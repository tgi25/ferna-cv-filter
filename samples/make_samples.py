"""Generate a sample job description and a batch of fictional CVs (PDF + DOCX), including a
corrupted PDF and an image-only "scanned" PDF, for testing Ferna CV Filter.

    pip install reportlab python-docx
    python samples/make_samples.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "cvs"

JD = """Job Title: Senior Data Scientist
Company: Lanka Analytics (Pvt) Ltd
Location: Colombo

We are looking for a Senior Data Scientist to build machine learning products for our retail and banking clients.

Requirements
- Bachelor's degree in Computer Science, Statistics or Mathematics or a related field.
- Minimum 4 years of experience in data science or machine learning roles.
- Strong skills in Python, SQL, machine learning, scikit-learn and data visualization.
- Experience with AWS and Docker for deploying models.
- AWS Certified Machine Learning Specialty certification is required.
- Relocation support available for candidates outside Colombo.

Nice to have
- A Master's degree is preferred.
- Experience with PyTorch, natural language processing and Spark is an advantage.
"""

CVS = {
    "Nimal_Perera_CV.docx": """NIMAL PERERA
Colombo, Sri Lanka | nimal.perera@example.com | +94 77 123 4567
Date of Birth: 14/03/1991

PROFESSIONAL SUMMARY
Data scientist with 7 years of experience delivering machine learning products for banking and retail.

WORK EXPERIENCE
Lead Data Scientist | Ceylon Bank PLC | Jan 2021 - Present
- Led a team of 6 data scientists building credit-risk models with Python and scikit-learn.
- Reduced loan default losses by 18% through a gradient boosting early-warning system.
Data Scientist | RetailHub Lanka | Mar 2018 - Dec 2020
- Built demand forecasting pipelines with SQL, Spark and AWS; deployed models with Docker.
Associate Data Analyst | DataWorks | Jun 2017 - Feb 2018

EDUCATION
MSc in Data Science, University of Moratuwa, 2020 - Distinction
BSc (Hons) in Statistics, University of Colombo, 2016 - First Class

SKILLS
Python, SQL, Machine Learning, scikit-learn, PyTorch, Natural Language Processing, Data Visualization, Tableau, AWS, Docker, Spark

CERTIFICATIONS
AWS Certified Machine Learning - Specialty (2022)

ACHIEVEMENTS
Gold medal for the best undergraduate research project, University of Colombo
Published a paper at the IEEE International Conference on Data Mining 2021
Employee of the Year 2022, Ceylon Bank PLC

EXTRA-CURRICULAR ACTIVITIES
Captain of the university cricket team (2014-2015)
Volunteer tutor, Rotaract Club of Colombo - weekend coding classes for school children
Member of the IEEE Computational Intelligence Society
""",
    "Ayesha_Fernando_Resume.docx": """Ayesha Fernando
Nugegoda, Colombo, Sri Lanka | ayesha.f@example.com | 071 234 5678

PROFILE
Analytical data scientist with 3 years of experience in machine learning for e-commerce.

EXPERIENCE
Data Scientist, ShopFast Lanka, Aug 2023 - Present
Junior Data Scientist, InsightLab, Sep 2022 - Jul 2023
- Collaborated with a cross-functional product team to build recommendation models.

EDUCATION
BSc in Computer Science, University of Sri Jayewardenepura, 2022 - Second Class (Upper)

SKILLS
Python, SQL, machine learning, scikit-learn, data visualization, Power BI, Docker

CERTIFICATIONS
AWS Certified Machine Learning - Specialty

AWARDS
Dean's List 2020 and 2021
Winner, SLIIT Codefest hackathon 2021

ACTIVITIES
Netball player, university team
Volunteer, blood donation campaign organiser
""",
    "Kasun_Silva_CV.pdf": """Kasun Silva
Kandy, Sri Lanka | kasun.silva@example.com | +94 76 555 1212
Willing to relocate to Colombo.

SUMMARY
Machine learning engineer with 5 years of experience in telecom analytics.

EMPLOYMENT HISTORY
Machine Learning Engineer - Dialog Analytics - 2019 - Present
- Built churn prediction models in Python and SQL; deployed with Docker on AWS.
- Mentored two interns and coordinated the analytics guild.

EDUCATION
BSc Engineering in Computer Engineering, University of Peradeniya, 2018

TECHNICAL SKILLS
Python, SQL, machine learning, scikit-learn, data visualization, AWS, Docker, Kubernetes

EXTRA-CURRICULAR
Member of the chess club; university chess team (2016)
""",
    "Ruwan_Jayasinghe_CV.pdf": """Ruwan Jayasinghe
Galle, Sri Lanka | ruwan.j@example.com | 077 999 8888
Not willing to relocate.

SUMMARY
Senior data scientist with 9 years of experience in insurance analytics.

PROFESSIONAL EXPERIENCE
Senior Data Scientist, Southern Insurance, 2017 - Present
Data Analyst, Galle Logistics, 2015 - 2017

EDUCATION
MSc in Statistics, University of Ruhuna, 2016
BSc in Mathematics, University of Ruhuna, 2014

SKILLS
Python, R, SQL, machine learning, scikit-learn, data visualization, AWS, Docker, Spark

CERTIFICATIONS
AWS Certified Machine Learning - Specialty

ACHIEVEMENTS
Increased claims fraud detection rate by 25% with an anomaly detection system
""",
    "Dilini_Wickramasinghe_CV.docx": """Dilini Wickramasinghe
Location: Colombo
dilini.w@example.com

EXPERIENCE
Data Scientist | FinServe Lanka | 2020 - 2025
- Developed customer segmentation models using Python, SQL and scikit-learn.
- Built dashboards for data visualization in Tableau.

EDUCATION
Higher National Diploma in Information Technology, SLIIT, 2019

SKILLS
Python, SQL, machine learning, scikit-learn, Tableau, data visualization, AWS, Docker

ACTIVITIES
Toastmasters club member; volunteer at a beach clean-up programme
""",
}


def make_docx(path: Path, text: str) -> None:
    import docx

    d = docx.Document()
    for line in text.strip().split("\n"):
        d.add_paragraph(line)
    d.save(path)


def make_pdf(path: Path, text: str) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    y = 800
    for line in text.strip().split("\n"):
        c.setFont("Helvetica", 10)
        c.drawString(50, y, line)
        y -= 14
        if y < 50:
            c.showPage()
            y = 800
    c.save()


def make_scanned_pdf(path: Path) -> None:
    """Image-only page (no text layer) — should be reported as 'needs OCR'."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    for i in range(30):
        c.rect(50, 780 - i * 20, 300 + (i % 5) * 40, 8, fill=1)
    c.save()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (HERE / "job_description.txt").write_text(JD, encoding="utf-8")
    for name, text in CVS.items():
        (make_pdf if name.endswith(".pdf") else make_docx)(OUT / name, text)
    (OUT / "corrupted_cv.pdf").write_bytes(b"%PDF-1.7\n1 0 obj << /Type /Catalog >>\n garbage \x00\x01\x02")
    make_scanned_pdf(OUT / "scanned_cv.pdf")
    print(f"Wrote job_description.txt and {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
