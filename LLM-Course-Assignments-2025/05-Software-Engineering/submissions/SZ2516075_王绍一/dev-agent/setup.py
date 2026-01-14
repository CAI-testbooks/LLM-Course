from setuptools import setup, find_packages

setup(
    name="dev-agent-mvp",
    version="0.1.0",
    packages=find_packages(where="src"),  # 在src目录中查找包
    package_dir={"": "src"},  # 告诉setuptools包都在src目录下
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "click>=8.0.0",
        "rich>=13.0.0",
    ],
    entry_points={
        "console_scripts": [
            "devagent=cli.main:cli",  # 注意：从src.cli改为cli
        ],
    },
    python_requires='>=3.8',
)