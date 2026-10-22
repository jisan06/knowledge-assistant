import os
from typing import List


def create_pdf_bytes(pages_content: List[List[str]]) -> bytes:
    """
    Constructs valid, multi-page PDF 1.4 bytes from a list of pages,
    where each page is a list of text lines.
    Uses standard Type 1 font (Helvetica) and standard PDF text operators.
    """
    objects = []
    
    # 1. Catalog
    catalog_obj = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    
    # 2. Pages object
    # Page object numbers will be: 3, 5, 7, 9 ...
    # Content object numbers will be: 4, 6, 8, 10 ...
    # Font object will be the last object: 2 + len(pages)*2 + 1
    num_pages = len(pages_content)
    font_obj_num = 2 + num_pages * 2 + 1
    
    kids_refs = " ".join([f"{3 + i * 2} 0 R" for i in range(num_pages)])
    pages_obj = f"2 0 obj\n<< /Type /Pages /Kids [{kids_refs}] /Count {num_pages} >>\nendobj\n".encode("latin-1")
    
    page_and_content_objs = []
    for i, lines in enumerate(pages_content):
        page_obj_num = 3 + i * 2
        content_obj_num = page_obj_num + 1
        
        # Build text stream
        ops = ["BT", "/F1 10 Tf", "50 730 Td", "14 TL"]
        for j, line in enumerate(lines):
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            if j == 0:
                ops.append(f"({safe}) Tj")
            else:
                ops.append(f"T* ({safe}) Tj")
        ops.append("ET")
        stream_data = "\n".join(ops).encode("latin-1")
        
        # Page object
        p_obj = (
            f"{page_obj_num} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_obj_num} 0 R "
            f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> >>\n"
            f"endobj\n"
        ).encode("latin-1")
        
        # Content stream object
        c_obj = (
            f"{content_obj_num} 0 obj\n"
            f"<< /Length {len(stream_data)} >>\n"
            f"stream\n"
        ).encode("latin-1") + stream_data + b"\nendstream\nendobj\n"
        
        page_and_content_objs.extend([p_obj, c_obj])
        
    font_obj = f"{font_obj_num} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode("latin-1")
    
    # Assemble all body objects
    all_objs = [catalog_obj, pages_obj] + page_and_content_objs + [font_obj]
    
    header = b"%PDF-1.4\n"
    
    # Calculate byte offsets for xref table
    offsets = []
    current_offset = len(header)
    for obj in all_objs:
        offsets.append(current_offset)
        current_offset += len(obj)
        
    xref_offset = current_offset
    total_objs = len(all_objs) + 1  # including object 0
    
    xref = f"xref\n0 {total_objs}\n0000000000 65535 f \n".encode("latin-1")
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode("latin-1")
        
    trailer = (
        f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin-1")
    
    return header + b"".join(all_objs) + xref + trailer


# Content for Acme Company Handbook across 4 distinct pages
HANDBOOK_PAGES = [
    # Page 1: Overview, Culture & Hybrid Schedule
    [
        "ACME CLOUD TECHNOLOGIES INC. - EMPLOYEE HANDBOOK",
        "Section 1: Company Overview, Culture and Workplace Policy",
        "",
        "Acme Cloud Technologies Inc. was founded in 2018 in Seattle, Washington.",
        "Our core mission is engineering distributed cloud infrastructure and automated AI knowledge tools.",
        "The company maintains headquarters in Seattle, WA and a regional engineering center in Austin, TX.",
        "",
        "Workplace Model and Schedule:",
        "Acme operates on a hybrid workplace model for all full-time employees.",
        "All team members are required to work on-site in the office on Tuesdays and Thursdays.",
        "Mondays, Wednesdays, and Fridays are designated flexible remote work days.",
        "",
        "Core Collaboration Hours:",
        "Regardless of physical location, all employees must be available online for team collaboration",
        "between 10:00 AM and 3:00 PM Pacific Standard Time (PST).",
        "A standard full-time work week consists of 40 scheduled hours.",
        "",
        "Communication Channels:",
        "Company-wide announcements are published in the #announcements Slack channel.",
        "Engineering design reviews and architecture proposals must be documented in Confluence.",
    ],
    # Page 2: Compensation, Payroll & Benefits
    [
        "ACME CLOUD TECHNOLOGIES INC. - EMPLOYEE HANDBOOK",
        "Section 2: Compensation, Payroll, Equity and Benefits",
        "",
        "Payroll Schedule:",
        "Employees are paid on a bi-weekly cycle on alternating Fridays via direct deposit.",
        "Electronic pay stubs and tax withholdings are accessible through the Gusto portal.",
        "",
        "Health and Medical Coverage:",
        "Comprehensive health, dental, and vision insurance is administered through BlueCross BlueShield.",
        "Acme covers 90 percent of premium costs for full-time employees and 70 percent for dependents.",
        "Coverage takes effect on the first calendar day of the month following the employee hire date.",
        "",
        "401(k) Retirement Savings:",
        "Our 401(k) retirement plan is managed by Fidelity Investments.",
        "Acme matches 100 percent of employee contributions up to 4 percent of annual base salary.",
        "All company matching funds vest immediately upon contribution.",
        "",
        "Annual Technology and Home Office Stipend:",
        "Full-time staff receive an annual stipend of 1,200 dollars per calendar year for home office expenses.",
        "Eligible expenses include external monitors, ergonomic chairs, desks, keyboards, and home internet.",
        "Receipts must be submitted through Expensify within 60 days of the transaction.",
        "",
        "Performance Bonus Structure:",
        "Annual discretionary bonuses target 10 to 15 percent of base salary and are paid in February.",
    ],
    # Page 3: Paid Time Off (PTO), Sick Leave & Parental Leave
    [
        "ACME CLOUD TECHNOLOGIES INC. - EMPLOYEE HANDBOOK",
        "Section 3: Paid Time Off, Leaves of Absence and Holidays",
        "",
        "Annual Vacation PTO:",
        "Full-time salaried staff accrue 20 vacation days (160 hours) of paid time off per calendar year.",
        "Vacation accrues at an approximate rate of 6.15 hours per bi-weekly pay period.",
        "",
        "PTO Rollover and Forfeiture Rules:",
        "Employees may roll over a maximum of 5 unused PTO days into the following calendar year.",
        "Any accrued vacation days exceeding the 5-day rollover limit are forfeited on December 31st.",
        "Forfeited days are not eligible for cash payout or retroactive extension.",
        "",
        "Dedicated Sick Leave:",
        "Employees receive 10 paid sick days annually, loaded into their balance on January 1st.",
        "Sick days are separate from vacation PTO and do not roll over from year to year.",
        "",
        "Paid Parental Leave:",
        "Acme offers 16 weeks of 100 percent paid parental leave following childbirth, adoption, or foster care.",
        "Both birthing and non-birthing parents are eligible after 6 months of continuous full-time tenure.",
        "",
        "Bereavement and Special Leave:",
        "Employees receive up to 5 consecutive paid business days for the loss of an immediate family member.",
        "Up to 2 weeks of paid jury duty leave is provided for civic jury summons.",
    ],
    # Page 4: Information Security, Hardware & Incident Response
    [
        "ACME CLOUD TECHNOLOGIES INC. - EMPLOYEE HANDBOOK",
        "Section 4: Information Security, Computing Equipment and Privacy",
        "",
        "Issued Computing Hardware:",
        "Engineering staff receive an Apple MacBook Pro 16-inch with M3 Pro chip and 36GB memory,",
        "or an enterprise Dell XPS 15 workstation. Hardware is refreshed on a 3-year replacement cycle.",
        "",
        "Password and Authentication Requirements:",
        "Passwords must be at least 16 characters in length and must be changed every 90 days.",
        "Multi-Factor Authentication (MFA) is strictly mandatory using hardware security keys or Okta Verify.",
        "SMS-based two-factor authentication is prohibited for internal company systems.",
        "",
        "Data Classification and Encryption Standards:",
        "Company data is classified into three tiers: Public, Internal Only, and Restricted Confidential.",
        "Customer personally identifiable information (PII) is classified as Restricted Confidential.",
        "All Restricted data must be encrypted at rest using AES-256 and in transit using TLS 1.3.",
        "",
        "Security Incident Reporting:",
        "Any lost device, suspicious email phishing attempt, or potential data breach must be reported",
        "within 1 hour of discovery to security@acmecloud.example.com or in Slack channel #security-alerts.",
        "",
        "Clean Desk and Screen Lock Policy:",
        "Computer screens must be locked whenever stepping away from the desk.",
    ],
]


def generate_handbook_pdf(output_path: str = None) -> str:
    """Generates company_handbook.pdf and saves to disk."""
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "data", "sample_documents")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "company_handbook.pdf")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    pdf_bytes = create_pdf_bytes(HANDBOOK_PAGES)
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)

    return output_path


if __name__ == "__main__":
    path = generate_handbook_pdf()
    print(f"Successfully generated sample PDF at: {path}")
    print(f"File size: {os.path.getsize(path)} bytes across {len(HANDBOOK_PAGES)} pages.")
