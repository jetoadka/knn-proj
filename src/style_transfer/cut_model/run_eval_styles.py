import os
import shutil
import subprocess

# --- SETTINGS ---
dataroot = "../data-pg"
testB_dir = os.path.join(dataroot, "testB")
styles_dir = "../reference_styles"  # Directory with 5 reference styles
output_base = "./results/conditional_cut_pg_200/test_latest/images" # Default CUT output path
final_results_dir = "./final_style_results" # Path to sort the final results

# Ensure final results directory exists
os.makedirs(final_results_dir, exist_ok=True)

# Get all reference styles (e.g., style_1.jpg)
style_images = [f for f in os.listdir(styles_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

for style_img in style_images:
    print(f"\n" + "="*40)
    print(f" Generating for style: {style_img} ")
    print("="*40)
    
    # 1. Clean testB and copy only the current style
    if os.path.exists(testB_dir):
        shutil.rmtree(testB_dir)
    os.makedirs(testB_dir)
    
    style_path = os.path.join(styles_dir, style_img)
    shutil.copy(style_path, os.path.join(testB_dir, style_img))
    
    # 2. Run test.py
    # Generate 25 images from testA
    command = [
        "python", "test.py",
        "--dataroot", dataroot,
        "--name", "conditional_cut_pg_200",
        "--model", "cut",
        "--gpu_ids", "0",
        "--num_test", "25",
        "--phase", "test"
    ]
    subprocess.run(command)
    
    # 3. Move and rename results
    style_name = os.path.splitext(style_img)[0]
    style_output_dir = os.path.join(final_results_dir, f"results_{style_name}")
    
    # Remove existing output dir if any
    if os.path.exists(style_output_dir):
        shutil.rmtree(style_output_dir)
        
    # Move results to the final directory
    if os.path.exists(output_base):
        shutil.move(output_base, style_output_dir)
        print(f">>> Results for {style_name} saved in: {style_output_dir}\n")
    else:
        print(f"!!! WARNING: Folder {output_base} not found.")

print("\n Done! All styles generated and sorted.")