import gradio as gr
import numpy as np

def awan(img):
    sepia_filter = np.array([
            [.001, .001, .001],
            [.001, .0, .001],
            [.001, .001, .001]])

    sepia_img = img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()

    output_txt = "You might be sick"
    return (sepia_img, output_txt)

awan = gr.Interface(
    fn = awan,
    inputs = gr.Image(label="Upload image or take photo here"),
    outputs = ["image", "text"], title="output image and analysis result",
    examples = ["8.JPG"],
    live = True,
    description = "Input image to get analysis"
).launch(share=True,debug=True, auth=("u", "p"), auth_message="Username is \"u\" and Password is \"p\"")