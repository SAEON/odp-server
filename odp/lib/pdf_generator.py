from dataclasses import dataclass
from io import BytesIO
from typing import Optional, List

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer, SimpleDocTemplate


@dataclass
class PersonInfo:
    """Person information (creator, contributor, contact)."""
    name: str = "N/A"
    affiliation: str = "N/A"
    email: str = "N/A"
    orcid: str = "N/A"


@dataclass
class GeographicExtent:
    """Geographic bounding box."""
    north: float
    south: float
    east: float
    west: float

    def __str__(self) -> str:
        """Format geographic extent as readable string."""
        return (
            f"North: {self.north}\n"
            f"South: {self.south}\n"
            f"West: {self.west}\n"
            f"East: {self.east}"
        )


@dataclass
class TemporalExtent:
    """Temporal extent (start and end dates)."""
    start_date: str = "N/A"
    end_date: str = "N/A"

    def __str__(self) -> str:
        """Format temporal extent as readable string."""
        return f"{self.start_date} – {self.end_date}"


@dataclass
class License:
    """License information."""
    text: str = "N/A"
    uri: str = ""

    def to_html(self) -> str:
        """Convert license to HTML link if URI available."""
        if self.uri:
            return f'<link href="{self.uri}">{self.text}</link>'
        return self.text


@dataclass
class RecordMetadata:
    """Unified internal representation of catalog record metadata.

    This dataclass normalizes metadata from different schemas (DataCite, ISO19115)
    into a single format that can be consistently processed for PDF generation.
    """
    # Required fields
    title: str
    doi: str
    publisher: str
    publication_year: str

    # Optional fields
    abstract: str = "N/A"
    keywords: List[str] = None
    creator: Optional[PersonInfo] = None
    contact: Optional[PersonInfo] = None
    license: Optional[License] = None
    geography: Optional[GeographicExtent] = None
    temporal: Optional[TemporalExtent] = None

    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.keywords is None:
            self.keywords = []
        if self.creator is None:
            self.creator = PersonInfo()
        if self.contact is None:
            self.contact = PersonInfo()
        if self.license is None:
            self.license = License()
        if self.temporal is None:
            self.temporal = TemporalExtent()


def generate_pdf(metadata: RecordMetadata) -> BytesIO:
    """Generate a metadata PDF from unified record metadata.

    This function creates a styled PDF table containing the record metadata.
    The PDF is generated in-memory and returned as a BytesIO buffer.

    Args:
        metadata: RecordMetadata object with normalized metadata

    Returns:
        BytesIO buffer containing the PDF binary data

    Raises:
        ValueError: If metadata is invalid or generation fails
        Exception: If ReportLab generation fails

    Example:
        >>> metadata = RecordMetadata(
        ...     title="Sample Dataset",
        ...     doi="10.15493/example",
        ...     publisher="SAEON",
        ...     publication_year="2024"
        ... )
        >>> pdf_buffer = generate_pdf(metadata)
        >>> pdf_bytes = pdf_buffer.getvalue()
    """
    try:
        # Setup PDF styles
        styles = getSampleStyleSheet()
        label_style = ParagraphStyle(
            "label",
            parent=styles["BodyText"],
            fontSize=10,
            leading=13,
            spaceAfter=0,
            spaceBefore=2,
            leftIndent=0,
            rightIndent=6,
            textColor=colors.black,
            wordWrap="LTR",
            bold=True,
        )
        value_style = ParagraphStyle(
            "value",
            parent=styles["BodyText"],
            fontSize=10,
            leading=13,
            spaceAfter=0,
            spaceBefore=2,
        )
        title_value_style = ParagraphStyle(
            "title_value",
            parent=value_style,
            fontSize=11,
            leading=14,
            spaceBefore=0,
            spaceAfter=2,
            bold=True,
        )

        # Build table rows with extracted data
        creator_text = f"{metadata.creator.name}<br/>{metadata.creator.affiliation}"
        if metadata.creator.email and metadata.creator.email != "N/A":
            creator_text += f"<br/>{metadata.creator.email}"

        contact_text = f"Contact Person: {metadata.contact.name}<br/>{metadata.contact.affiliation}"
        if metadata.contact.email and metadata.contact.email != "N/A":
            contact_text += f"<br/>email: {metadata.contact.email}"

        rows = [
            [
                Paragraph("Title", label_style),
                Paragraph(metadata.title, title_value_style),
            ],
            [
                Paragraph("DOI", label_style),
                Paragraph(
                    f'<link href="https://doi.org/{metadata.doi}">https://doi.org/{metadata.doi}</link>',
                    value_style,
                ),
            ],
            [
                Paragraph("Authors", label_style),
                Paragraph(creator_text, value_style),
            ],
            [
                Paragraph("Publisher", label_style),
                Paragraph(
                    f"{metadata.publisher} ({metadata.publication_year})",
                    value_style,
                ),
            ],
            [
                Paragraph("Contributors", label_style),
                Paragraph(contact_text, value_style),
            ],
            [
                Paragraph("Abstract", label_style),
                Paragraph(metadata.abstract, value_style),
            ],
            [
                Paragraph("Licence", label_style),
                Paragraph(metadata.license.to_html(), value_style),
            ],
            [
                Paragraph("Temporal extent", label_style),
                Paragraph(str(metadata.temporal), value_style),
            ],
            [
                Paragraph("Geographic extent", label_style),
                Paragraph(str(metadata.geography).replace("\n", "<br/>") if metadata.geography else "N/A", value_style),
            ],
            [
                Paragraph("Keywords", label_style),
                Paragraph(", ".join(metadata.keywords) if metadata.keywords else "N/A", value_style),
            ],
        ]

        # Create table
        table = Table(
            rows,
            colWidths=[1.6 * inch, 5.3 * inch],
            hAlign="LEFT",
            repeatRows=0,
        )

        # Table styling
        tbl_style = [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, 0), 0.25, colors.lightgrey),
        ]
        for r in range(1, len(rows)):
            tbl_style.append(("LINEBELOW", (0, r), (-1, r), 0.25, colors.lightgrey))

        table.setStyle(TableStyle(tbl_style))

        # Build PDF document
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=(8.5 * inch, 11 * inch),
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        story = [table, Spacer(1, 0.2 * inch)]
        doc.build(story)
        buffer.seek(0)
        return buffer

    except Exception as e:
        raise ValueError(f"PDF generation failed: {str(e)}") from e
