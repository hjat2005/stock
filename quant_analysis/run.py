#!/usr/bin/env python3
"""
Quant Analysis System 启动脚本
"""

import os
import sys
import argparse

def init_system():
    """初始化系统"""
    print("🚀 初始化量化分析系统...")
    
    # 初始化数据库
    from models import init_db
    init_db()
    
    print("✅ 系统初始化完成！")

def run_web():
    """启动Web服务"""
    import subprocess
    
    web_path = os.path.join(os.path.dirname(__file__), 'web', 'app.py')
    
    print("🌐 启动Web服务...")
    print("📍 访问地址: http://localhost:8501")
    print("\n按 Ctrl+C 停止服务\n")
    
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', web_path,
        '--server.port', '8501',
        '--server.address', '0.0.0.0'
    ])

def run_tests():
    """运行测试"""
    import subprocess
    
    print("🧪 运行测试...")
    subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'])

def main():
    parser = argparse.ArgumentParser(description='Quant Analysis System')
    parser.add_argument('command', choices=['init', 'web', 'test'], 
                       help='命令: init (初始化), web (启动Web), test (运行测试)')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        init_system()
    elif args.command == 'web':
        init_system()
        run_web()
    elif args.command == 'test':
        run_tests()

if __name__ == '__main__':
    main()
