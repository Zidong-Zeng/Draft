import platform
import os
import subprocess

def detect_os():
    """检测操作系统类型"""
    os_type = platform.system().lower()
    if os_type == 'windows':
        return 'windows'
    elif os_type == 'linux':
        return 'linux'
    elif os_type == 'darwin':
        return 'macos'
    else:
        return 'unknown'

def set_hf_endpoint():
    """设置HF_ENDPOINT环境变量"""
    os_type = detect_os()
    if os_type == 'linux' or os_type == 'macos':
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        # 建议用户写入~/.bashrc
        print("建议将以下内容添加到 ~/.bashrc 文件中：")
        print("export HF_ENDPOINT=https://hf-mirror.com")
    elif os_type == 'windows':
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("建议将以下内容添加到系统环境变量中：")
        print("HF_ENDPOINT=https://hf-mirror.com")

def download_huggingface():
    """主函数：获取用户输入并执行下载命令"""
    print("欢迎使用HuggingFace下载工具")
    print("当前操作系统:", detect_os().capitalize())
    
    # 设置HF_ENDPOINT
    set_hf_endpoint()
    
    # 获取用户输入
    download_type = input("请输入下载类型（model/dataset）: ").strip().lower()
    while download_type not in ['model', 'dataset']:
        print("输入错误，请输入'model'或'dataset'")
        download_type = input("请输入下载类型（model/dataset）: ").strip().lower()
    
    name = input(f"请输入{download_type}名称: ").strip()
    local_dir = input("请输入下载到的文件夹路径: ").strip()
    token = input("请输入token（可选，直接回车跳过）: ").strip()
    
    # 构建命令
    cmd = ["huggingface-cli", "download", "--resume-download"]
    
    if download_type == 'dataset':
        cmd.append("--repo-type")
        cmd.append("dataset")
    
    if token:
        cmd.append("--token")
        cmd.append(token)
    
    cmd.append(name)
    cmd.append("--local-dir")
    cmd.append(local_dir)
    
    # 执行命令
    print("\n即将执行以下命令:")
    print(" ".join(cmd))
    print("\n开始下载...")
    
    try:
        subprocess.run(cmd, check=True)
        print("下载完成！")
    except subprocess.CalledProcessError as e:
        print(f"下载失败: {e}")
    except FileNotFoundError:
        print("错误: huggingface-cli 未安装，请先运行 'pip install -U huggingface_hub'")

if __name__ == "__main__":
    download_huggingface()
