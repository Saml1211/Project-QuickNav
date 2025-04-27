from setuptools import setup, find_packages
import os

# Read version
def get_version():
    version_path = os.path.join(os.path.dirname(__file__), "VERSION.txt")
    with open(version_path, "r") as vfile:
        return vfile.read().strip()

setup(
    name="quicknav",
    version=get_version(),
    description="Project QuickNav: 5-digit project code navigation utility with MCP server integration",
    long_description=open("README.md", encoding="utf-8").read() if os.path.isfile("README.md") else "",
    long_description_content_type="text/markdown",
    author="Pro AV Solutions",
    packages=find_packages(include=["quicknav", "quicknav.*", "mcp_server", "mcp_server.*"]),
    install_requires=[
        "mcp[cli]",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "": ["VERSION.txt"],
    },
    entry_points={
        "console_scripts": [
            "quicknav=quicknav.cli:main",
            "quicknav-mcp=mcp_server.__main__:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "License :: OSI Approved :: MIT License",
    ],
    project_urls={
        "Source": "https://github.com/your-org/quicknav",
        "Documentation": "https://github.com/your-org/quicknav",
        "Tracker": "https://github.com/your-org/quicknav/issues",
    },
    zip_safe=False,
)