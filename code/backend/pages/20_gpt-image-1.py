import os
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
api_key = os.getenv("IMG_API_KEY")
azure_endpoint = os.getenv("IMG_AZURE_ENDPOINT")
api_version = os.getenv("IMG_API_VERSION")

azure_base_url = f"{azure_endpoint}/openai/deployments/gpt-image-1"
HEADERS = {"api-key": api_key}


def generate_image(prompt, size="1024x1024", quality="auto", n=1):
    url = f"{azure_base_url}/images/generations?api-version={api_version}"
    payload = {"prompt": prompt, "size": size, "quality": quality, "n": n}
    response = requests.post(
        url, headers={**HEADERS, "Content-Type": "application/json"}, json=payload
    )
    response.raise_for_status()
    results = []
    for item in response.json()["data"]:
        b64_image = item["b64_json"]
        image_bytes = base64.b64decode(b64_image)
        results.append(Image.open(BytesIO(image_bytes)))
    return results


def edit_image(image_file, prompt, mask_file=None, size="1024x1024", quality="auto"):
    url = f"{azure_base_url}/images/edits?api-version={api_version}"
    files = {
        "image": ("input.png", image_file, "image/png"),
        "prompt": (None, prompt),
        "size": (None, size),
        "quality": (None, quality),
        "n": (None, "1"),
    }
    if mask_file:
        files["mask"] = mask_file
    response = requests.post(url, headers=HEADERS, files=files)
    response.raise_for_status()
    b64_image = response.json()["data"][0]["b64_json"]
    image_bytes = base64.b64decode(b64_image)
    return Image.open(BytesIO(image_bytes))


def generate_mask(image_file):
    url = f"{azure_base_url}/images/edits?api-version={api_version}"
    prompt = (
        "generate a mask delimiting the entire character in the picture, "
        "using white where the character is and black for the background. "
        "Return an image in the same size as the input image."
    )
    files = {
        "image": ("input.png", image_file, "image/png"),
        "prompt": (None, prompt),
        "quality": (None, "medium"),
    }
    response = requests.post(url, headers=HEADERS, files=files)
    response.raise_for_status()
    b64_image = response.json()["data"][0]["b64_json"]
    image_bytes = base64.b64decode(b64_image)
    return Image.open(BytesIO(image_bytes))


def add_alpha_channel(mask_img):
    mask_gray = mask_img.convert("L")
    threshold = 128
    mask_bw = mask_gray.point(lambda x: 255 if x > threshold else 0, mode="1")
    mask_rgba = mask_bw.convert("RGBA")
    mask_alpha = mask_bw.convert("L")
    mask_rgba.putalpha(mask_alpha)
    buf = BytesIO()
    mask_rgba.save(buf, format="PNG")
    buf.seek(0)
    return buf


def download_image(image, filename="result.png"):
    buf = BytesIO()
    image.save(buf, format="PNG")
    byte_data = buf.getvalue()
    st.download_button(
        "💾 下载图片", data=byte_data, file_name=filename, mime="image/png"
    )


# 页面设置
st.set_page_config(page_title="AI 图像平台", layout="wide")
st.title("🎨 gpt-image-1 图像生成与编辑")

mode = st.radio(
    "选择功能",
    ["🔮 文本生成图片", "✏️ 编辑上传图片", "🎭 Mask编辑上传图片"],
    horizontal=True,
)

if mode == "🔮 文本生成图片":
    st.markdown("### 🔮 文本生成图片")

    col1, col2, col3 = st.columns(3)
    with col1:
        size = st.radio(
            "📐 尺寸",
            ["auto", "1024x1024", "1536x1024", "1024x1536"],
            horizontal=True,
            key="gen_size",
        )
    with col2:
        quality = st.radio(
            "🖼️ 质量",
            ["auto", "low", "medium", "high"],
            horizontal=True,
            key="gen_quality",
        )
    with col3:
        count = st.radio("🖼️ 数量", [1, 2, 3, 4], horizontal=True, key="gen_count")

    prompt = st.text_area(
        "✏️ 输入 Prompt",
        placeholder="如：一只穿着西装的可爱小狗在咖啡馆喝咖啡",
        height=100,
        key="gen_prompt",
    )

    if st.button("🚀 开始生成", ):
        if prompt.strip():
            with st.spinner("生成中..."):
                try:
                    images = generate_image(prompt, size=size, quality=quality, n=count)
                    st.session_state["text2img_results"] = images
                    st.success("生成成功！")
                except Exception as e:
                    st.error(f"生成失败: {e}")
        else:
            st.warning("请输入 Prompt")

    if "text2img_results" in st.session_state and st.session_state["text2img_results"]:
        st.markdown("### 🖼️ 生成结果")
        st.markdown(
            '<div style="display:flex; overflow-x:auto; gap:20px; flex-wrap: nowrap;">',
            unsafe_allow_html=True,
        )
        for idx, img in enumerate(st.session_state["text2img_results"]):
            with BytesIO() as output:
                img.save(output, format="PNG")
                img_data = output.getvalue()
                b64 = base64.b64encode(img_data).decode()
                st.markdown(
                    f"""
                    <div style="flex:0 0 auto; border:1px solid #ddd; padding:10px; display:inline-block; margin-right:20px; border-radius:8px;">
                        <img src="data:image/png;base64,{b64}" width="{img.width}" height="{img.height}" style="border-radius:5px;" />
                        <a download="generated_{idx+1}.png" href="data:image/png;base64,{b64}">
                            <button style="width:100%; margin-top:8px;">💾 下载图片 {idx+1}</button>
                        </a>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)


elif mode == "✏️ 编辑上传图片":
    st.markdown("### ✏️ 上传图片并编辑")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "上传图片", type=["jpg", "jpeg", "png"], key="edit_upload"
        )
    with col2:
        size = st.radio(
            "📐 尺寸",
            ["auto", "1024x1024", "1536x1024", "1024x1536"],
            key="edit_size",
            horizontal=True,
        )
    with col3:
        quality = st.radio(
            "🖼️ 质量",
            ["auto", "low", "medium", "high"],
            key="edit_quality",
            horizontal=True,
        )

    if uploaded_file:
        st.image(uploaded_file, caption="原始图片", )

    edit_prompt = st.text_area("📝 编辑描述 (Prompt)", key="edit_prompt")
    if st.button("🚀 开始编辑", ):
        if uploaded_file and edit_prompt.strip():
            with st.spinner("编辑中..."):
                try:
                    result = edit_image(
                        uploaded_file,
                        edit_prompt,
                        size=size,
                        quality=quality,
                    )
                    st.session_state["edited_image_tab2"] = result
                    st.success("编辑完成！")
                except Exception as e:
                    st.error(f"编辑失败: {e}")
        else:
            st.warning("请上传图片并填写 Prompt")

    if "edited_image_tab2" in st.session_state:
        st.image(
            st.session_state["edited_image_tab2"],
            caption="编辑后的图片",
        )
        download_image(st.session_state["edited_image_tab2"], "edited.png")

elif mode == "🎭 Mask编辑上传图片":
    st.markdown("### 🎭 自动生成 Mask 并编辑")

    if "mask_input_image" not in st.session_state:
        st.session_state["mask_input_image"] = None

    mask_image = st.file_uploader(
        "上传原图", type=["jpg", "jpeg", "png"], key="mask_upload"
    )

    if mask_image:
        st.session_state["mask_input_image"] = mask_image

    if st.session_state["mask_input_image"]:
        st.image(st.session_state["mask_input_image"], caption="原始图片", )

        if st.button("🛠️ 生成Mask", ):
            with st.spinner("生成中..."):
                try:
                    mask_img = generate_mask(st.session_state["mask_input_image"])
                    st.session_state["mask_img"] = mask_img
                    st.session_state["mask_generated"] = True
                    st.success("Mask生成成功！")
                except Exception as e:
                    st.error(f"生成失败: {e}")
                    st.session_state["mask_generated"] = False

    if st.session_state.get("mask_generated", False):
        st.image(
            st.session_state["mask_img"],
            caption="生成的Mask",
        )
        mask_prompt = st.text_area("📝 编辑描述 (Prompt)", key="mask_prompt")
        col1, col2 = st.columns(2)
        with col1:
            size = st.radio(
                "📐 输出尺寸",
                ["auto", "1024x1024", "1536x1024", "1024x1536"],
                key="mask_size",
                horizontal=True,
            )
        with col2:
            quality = st.radio(
                "🖼️ 图像质量",
                ["auto", "low", "medium", "high"],
                key="mask_quality",
                horizontal=True,
            )

        if st.button("🚀 使用Mask编辑图片", ):
            if mask_prompt.strip():
                with st.spinner("编辑中..."):
                    try:
                        mask_buf = add_alpha_channel(st.session_state["mask_img"])
                        image = edit_image(
                            st.session_state["mask_input_image"],
                            mask_prompt,
                            mask_file=("mask.png", mask_buf, "image/png"),
                            size=size,
                            quality=quality,
                        )
                        st.session_state["tab3_edited_image"] = image
                        st.success("编辑完成！")
                    except Exception as e:
                        st.error(f"编辑失败: {e}")
            else:
                st.warning("请填写 Prompt")

    if "tab3_edited_image" in st.session_state:
        st.markdown("### 🖼️ 编辑结果")
        with BytesIO() as output:
            st.session_state["tab3_edited_image"].save(output, format="PNG")
            img_data = output.getvalue()
            b64 = base64.b64encode(img_data).decode()
            st.markdown(
                f"""
                <div style="display:flex; overflow-x:auto; gap:20px;">
                    <div style="flex:0 0 auto; border:1px solid #ddd; padding:10px; border-radius:8px; min-width:250px;">
                        <img src="data:image/png;base64,{b64}" style="width:100%;" />
                        <a download="masked_edit.png" href="data:image/png;base64,{b64}">
                            <button style="width:100%; margin-top:8px;">💾 下载编辑图片</button>
                        </a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
