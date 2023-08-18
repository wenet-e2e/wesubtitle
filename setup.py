from setuptools import setup, find_packages

requirements = [
    "paddlepaddle",
    "paddleocr>=2.0.1",
    "opencv-python",
    "srt",
]

setup(
    name="wesubtitle",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={"console_scripts": [
        "wesubtitle = wesubtitle.main:main",
    ]},
)
