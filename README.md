# Resume-Analyzer
This project is a tool that analyzes resumes/CVs/Cover Letters. When used, it will take a resume/CV/cover letter and a job title as input, and it will score the compatibility between the file given and the job title. It will also interperet this compatibility as "poor," "decent," or "great."

# Installation and Usage
In order to use this program, you will need Python version 3 or greater and Pip installed.

After downloading main.py, you will need to run the following commands using Pip to use the project:

```bash
pip install torch
pip install transformers
pip install scipy
pip install docx2txt
```

Afterwards, you can run the project with:
```bash
python resume_analyzer.py
```

Once it's running, the program will prompt you for the job title/position and the filepath of a resume/CV/cover letter. Only .txt or .docx files are accepted for the file.
