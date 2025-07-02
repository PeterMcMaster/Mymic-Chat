import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("PeterMcMaster999/Mymic")
tokenizer = AutoTokenizer.from_pretrained("PeterMcMaster999/Mymic")

# Ensure the tokenizer has a padding token
tokenizer.pad_token = tokenizer.eos_token

# Response generation function
def generate_response(prompt):
    full_input = f"Friend: {prompt.strip()}\nYou:"
    
    input_ids = tokenizer(full_input, return_tensors="pt").input_ids
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.95,
        temperature=0.7
    )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Grab the first 2 lines after "You:"
    if "You:" in decoded:
        after_you = decoded.split("You:")[-1].strip()
        lines = [line.strip() for line in after_you.split("\n") if line.strip()]
        return "\n".join(lines[:2])
    else:
        return decoded.strip()

# Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=4, label="You:"),
    outputs=gr.Textbox(label="Mymic:"),
    title="Mymic",
    description="Custom GPT-2 chat bot fine-tuned on my personal message history."
)

if __name__ == "__main__":
    iface.launch()