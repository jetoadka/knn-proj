import os
import shutil
import subprocess
import glob

# --- SETTINGS ---
dataroot = "../data"
testB_dir = os.path.join(dataroot, "testB")
styles_dir = "../reference_styles"  
output_base = "./results/conditional_model_v2/test_4/images" 
final_results_dir = "./final_style_results" 

os.makedirs(final_results_dir, exist_ok=True)
style_images = [f for f in os.listdir(styles_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# --- HTML INITIALIZATION ---
html_content = """
<!DOCTYPE html>
<html lang="sk">
<head>
    <meta charset="UTF-8">
    <title>Style Transfer Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; color: #333; }
        h1 { text-align: center; }
        h2 { border-bottom: 2px solid #ccc; padding-bottom: 10px; margin-top: 50px; text-align: center; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; background-color: #fff; }
        th, td { border: 1px solid #ddd; padding: 15px; text-align: center; vertical-align: middle; }
        th { background-color: #e2e2e2; font-size: 1.1em; }
        img { max-width: 256px; height: auto; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <h1>Compare reference style with generated one</h1>
"""

for style_img in style_images:
    print(f"\n" + "="*40)
    print(f" Generating for style: {style_img} ")
    print("="*40)
    
    if os.path.exists(testB_dir):
        shutil.rmtree(testB_dir)
    os.makedirs(testB_dir)
    
    style_path = os.path.join(styles_dir, style_img)
    shutil.copy(style_path, os.path.join(testB_dir, style_img))
    
    # Run test.py
    command = [
        "python", "test.py",
        "--dataroot", dataroot,
        "--name", "conditional_model_v2",
        "--model", "cut",
        "--netG", "resnet_cat",
        "--nz", "256",
        "--ngf", "128",
        "--ndf", "128",
        "--dataset_mode", "unaligned",
        "--gpu_ids", "0",
        "--num_test", "25",
        "--phase", "test",
        "--epoch", "4",
        "--eval"
    ]
    subprocess.run(command)
    
    style_name = os.path.splitext(style_img)[0]
    style_output_dir = os.path.join(final_results_dir, f"results_{style_name}")
    
    if os.path.exists(style_output_dir):
        shutil.rmtree(style_output_dir)
        
    if os.path.exists(output_base):
        shutil.move(output_base, style_output_dir)
        print(f">>> Results for {style_name} saved in: {style_output_dir}\n")
            
        html_content += f"""
        <h2>Reference Style: {style_img}</h2>
        <table>
            <tr>
                <th>Original (real_A)</th>
                <th>Generated (fake_B)</th>
                <th>reference style (real_B)</th>
            </tr>
        """
        
        # Searching in subfolders
        fake_dir = os.path.join(style_output_dir, "fake_B")
        real_a_dir = os.path.join(style_output_dir, "real_A")
        real_b_dir = os.path.join(style_output_dir, "real_B")
        
        if os.path.exists(fake_dir):
            # Find all images in foulder fake_B
            fake_images = sorted(glob.glob(os.path.join(fake_dir, "*.*")))
            
            for fake_path in fake_images:
                filename = os.path.basename(fake_path)
                
                # Paths for the same picture in other folders
                real_a_path = os.path.join(real_a_dir, filename)
                real_b_path = os.path.join(real_b_dir, filename)
                
                if os.path.exists(real_a_path) and os.path.exists(real_b_path):
                    rel_real_a = f"results_{style_name}/real_A/{filename}"
                    rel_fake = f"results_{style_name}/fake_B/{filename}"
                    rel_real_b = f"results_{style_name}/real_B/{filename}"
                    
                    html_content += f"""
                    <tr>
                        <td><img src="{rel_real_a}" alt="Original"></td>
                        <td><img src="{rel_fake}" alt="Generated"></td>
                        <td><img src="{rel_real_b}" alt="Reference style"></td>
                    </tr>
                    """
        else:
            print(f"!!! WARNING: Folder {fake_dir} not found.")
        
        html_content += "</table>\n"
    else:
        print(f"!!! WARNING: Folder {output_base} not found.")

html_content += """
</body>
</html>
"""

html_file_path = os.path.join(final_results_dir, "index.html")
with open(html_file_path, "w", encoding="utf-8") as html_file:
    html_file.write(html_content)

print(f"\n>>> List and HTML visualization was saved to: {html_file_path}")