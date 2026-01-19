"""
清理 Streamlit 缓存和 Python 缓存
运行方式: python clear_cache.py
"""
import os
import shutil
import sys

def clear_cache():
    print("🧹 开始清理缓存...\n")
    
    # 1. 清理项目目录下的 __pycache__
    project_dir = os.path.dirname(os.path.abspath(__file__))
    pycache_dirs = []
    for root, dirs, files in os.walk(project_dir):
        for d in dirs:
            if d == "__pycache__":
                pycache_dirs.append(os.path.join(root, d))
    
    for pycache in pycache_dirs:
        try:
            shutil.rmtree(pycache)
            print(f"✅ 已删除: {pycache}")
        except Exception as e:
            print(f"❌ 删除失败 {pycache}: {e}")
    
    # 2. 清理用户目录下的 Streamlit 缓存
    user_home = os.path.expanduser("~")
    streamlit_cache_paths = [
        os.path.join(user_home, ".streamlit", "cache"),
        os.path.join(user_home, ".streamlit", "credentials.toml"),
    ]
    
    for cache_path in streamlit_cache_paths:
        if os.path.exists(cache_path):
            try:
                if os.path.isdir(cache_path):
                    shutil.rmtree(cache_path)
                else:
                    os.remove(cache_path)
                print(f"✅ 已删除: {cache_path}")
            except Exception as e:
                print(f"❌ 删除失败 {cache_path}: {e}")
        else:
            print(f"⏭️ 不存在: {cache_path}")
    
    # 3. 清理项目目录下的 .streamlit 缓存
    project_streamlit = os.path.join(project_dir, ".streamlit", "cache")
    if os.path.exists(project_streamlit):
        try:
            shutil.rmtree(project_streamlit)
            print(f"✅ 已删除: {project_streamlit}")
        except Exception as e:
            print(f"❌ 删除失败 {project_streamlit}: {e}")
    
    print("\n" + "="*50)
    print("🎉 缓存清理完成！")
    print("="*50)
    print("\n📌 接下来请执行以下步骤：")
    print("1. 重新启动 Streamlit:")
    print("   python -m streamlit run sqlagent\\web_ui.py")
    print("")
    print("2. 用新的浏览器标签页打开，或按 Ctrl+Shift+R 强制刷新")
    print("")

if __name__ == "__main__":
    clear_cache()
