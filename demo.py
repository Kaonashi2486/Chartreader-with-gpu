'''Changes:-
1)Removed shebang line (#!/root/miniconda3/envs/ChartLLM/bin/python) since it's Linux-specific and not needed on Windows.
2)Path adjustments: Changed paths from Linux to Windows format, such as C:\\path\\to\\your\\....
3)Subprocess changes: In the subprocess.run, replaced "python3" with "python" for Windows compatibility and adjusted the path separators (\\).'''

import gradio as gr
import os
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration
)
from ocr import Ocr
from PIL import Image
import subprocess
import json
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

SHOW_TEXT = False

# Set command and parameters
extraction_command = "val_extraction.py"
extraction_params = [
    "--save_path", "evaluation",
    "--model_type", "KPGrouping",
    "--cache_path", "C:\\path\\to\\your\\cache\\",  # Change this to your Windows path
    "--data_dir", "./",
    "--trained_model_iter", "best_full"
]

# Load tokenizer and model
t5_tokenizer = AutoTokenizer.from_pretrained("C:\\path\\to\\your\\t5_output\\checkpoint-70000")  # Update path for Windows
t5_model = T5ForConditionalGeneration.from_pretrained("C:\\path\\to\\your\\t5_output\\checkpoint-70000")  # Update path for Windows

def Visualization(im):
    file_name = os.getcwd() + "\\evaluation\\KPGroupingbest_full.json"  # Adjust path for Windows
    with open(file_name) as f:
        data = json.load(f)
    thres = 0.4
    chart_type = None
    filename = "test_img.png"
    im = np.array(Image.open(filename), dtype=np.uint8)
    fig, ax = plt.subplots(1, dpi=300)
    ax.imshow(im)
    key_point_color = 'skyblue'  # Key point color: sky blue
    center_point_color = 'coral'  # Center point color: coral
    group_line_color = 'lightgrey'
    chart_type = None

    groups = data[filename][2]
    sorted_groups = sorted(groups, key=lambda group: group[0])
    indexes_to_remove = set()

    # Identify elements to remove
    for i in range(len(sorted_groups) - 1):
        current_group = sorted_groups[i]
        next_group = sorted_groups[i + 1]
    
        if abs(current_group[0] - next_group[0]) < 5 and abs(current_group[1] - next_group[1]) < 5:
            if current_group[1] < next_group[1]:
                indexes_to_remove.add(i)
            else:
                indexes_to_remove.add(i + 1)

    # Remove marked elements
    for index in sorted(indexes_to_remove, reverse=True):
        del sorted_groups[index]
    
    print(f"sorted_groups:")
    for group in sorted_groups:
        print(group)
        category = group[-1]
        cen_in_group = group[0:2]
        print(f"cen_in_group {cen_in_group}")
        for k in range(1, len(group[:-1]) // 2):
            key_in_group = group[2 * k:2 * k + 2]
            print(f"key_in_group {key_in_group}")
            ax.plot([cen_in_group[0], key_in_group[0]], [cen_in_group[1], key_in_group[1]], color=group_line_color, linewidth=1)
            circle = patches.Circle((key_in_group[0], key_in_group[1]), radius=5, color=key_point_color)
            if SHOW_TEXT:
                ax.text(key_in_group[0], key_in_group[1] - 5, f'{key_in_group[0]:.2f}, {key_in_group[1]:.2f}', ha='center', va='top', fontsize=4, color='black')
            ax.add_patch(circle)
            circle = patches.Circle((cen_in_group[0], cen_in_group[1]), radius=6, color=center_point_color)
            ax.add_patch(circle)
            if SHOW_TEXT:
                ax.text(cen_in_group[0], cen_in_group[1] - 5, f'{cen_in_group[0]:.2f}, {cen_in_group[1]:.2f}', ha='center', va='top', fontsize=4, color='black')

    plt.axis('off')  # Hide axes
    plt.savefig('plot.png', dpi=300)
    return Image.open('plot.png'), chart_type

def ChartQA(chart_img, question, model):
    chart_img.save("test_img.png")
    chart_img.save(".\\images\\val\\test_img.png")
    subprocess.run(["python", os.getcwd() + "\\" + extraction_command] + extraction_params)  # Adjust for Windows
    visualized_img, chart_type = Visualization(chart_img)
    chart_table = Ocr(chart_type)
    answer = "ERROR"
    if model == "T5":
        question = "Question: " + question
        chart_info = "Table: " + chart_table
        query = question + " " + chart_info
        print(query)
        input_ids = t5_tokenizer(query, return_tensors="pt").input_ids
        outputs = t5_model.generate(input_ids, max_new_tokens=1000)
        answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        answer = "ERROR: Model not supported yet."
    return visualized_img, chart_table, answer

# Gradio interface
title = "ChartLLM Demo v1.0"
description = "This is the **description** of ChartEQA Demo v1.0"
example_bar = ["example_bar.png", "What is the value of light slate?", "T5"]
example_line = ["example_line.png", "Is the population of bears in 2019 greater than the population of bears in 2017?", "T5"]
example_line_2 = ["example_line.png", "Is the population of bears in 2017 greater than in 2022?", "T5"]

demo = gr.Interface(
    fn=ChartQA,
    inputs=[
        gr.Image(type="pil", label="Chart", info="Please upload your chart image here."),
        gr.Textbox(label="Query", info="Please input your query here."),
        gr.Radio(["T5"], label="Model", info="Please choose the model you want to use.")
    ],
    outputs=[
        gr.Image(label="Detected Key Points (Blue) and Center Points (Orange)", show_label=True),
        gr.Textbox(label="Converted Table", show_label=True),
        gr.Textbox(label="Answer", show_label=True).style(show_copy_button=True),
    ],
    examples=[example_bar, example_line, example_line_2],
    cache_examples=False,
    allow_flagging="never",
    title=title,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=True, server_port=6006)
