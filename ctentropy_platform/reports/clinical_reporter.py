"""
Clinical-Grade Reporting System
Generates professional medical reports for CTEntropy analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass, asdict
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    print("Warning: reportlab not available. PDF reports disabled.")
    REPORTLAB_AVAILABLE = False
    # Mock classes for when reportlab is not available
    class MockReportlab:
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return MockReportlab
    
    SimpleDocTemplate = MockReportlab
    Paragraph = MockReportlab
    getSampleStyleSheet = lambda: {'Normal': MockReportlab(), 'Heading1': MockReportlab(), 'Heading2': MockReportlab()}


@dataclass
class DiagnosticResult:
    """Structured diagnostic result for clinical reporting"""
    patient_id: str
    test_date: datetime
    condition: str
    confidence: float
    entropy_signature: Dict[str, float]
    risk_level: str
    recommendations: List[str]
    technical_details: Dict[str, Any]

@dataclass
class ClinicalMetadata:
    """Clinical metadata for report generation"""
    facility_name: str = "CTEntropy Diagnostic Center"
    physician_name: str = "Dr. [Physician Name]"
    technician_name: str = "CTEntropy System"
    report_version: str = "1.0"
    software_version: str = "CTEntropy v1.0"
    certification: str = "Research Use Only"

class ClinicalReporter:
    """Generate clinical-grade diagnostic reports"""
    
    def __init__(self, metadata: Optional[ClinicalMetadata] = None):
        self.metadata = metadata or ClinicalMetadata()
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom styles for clinical reports"""
        
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        )
        
        # Header style
        self.header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        # Clinical finding style
        self.finding_style = ParagraphStyle(
            'ClinicalFinding',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            leftIndent=20
        )
        
        # Warning style
        self.warning_style = ParagraphStyle(
            'Warning',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.red,
            spaceAfter=6
        )
    
    def generate_diagnostic_report(self, result: DiagnosticResult, 
                                 output_path: str = None) -> str:
        """
        Generate comprehensive clinical diagnostic report
        
        Args:
            result: Diagnostic result data
            output_path: Path to save PDF report
            
        Returns:
            Path to generated report
        """
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reports/CTEntropy_Report_{result.patient_id}_{timestamp}.pdf"
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Report header
        story.extend(self._create_header(result))
        
        # Patient information
        story.extend(self._create_patient_info(result))
        
        # Diagnostic summary
        story.extend(self._create_diagnostic_summary(result))
        
        # Detailed analysis
        story.extend(self._create_detailed_analysis(result))
        
        # Visualizations
        story.extend(self._create_visualizations(result))
        
        # Clinical recommendations
        story.extend(self._create_recommendations(result))
        
        # Technical details
        story.extend(self._create_technical_details(result))
        
        # Footer
        story.extend(self._create_footer())
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    def _create_header(self, result: DiagnosticResult) -> List:
        """Create report header"""
        
        elements = []
        
        # Title
        title = Paragraph("CTEntropy Neurological Diagnostic Report", self.title_style)
        elements.append(title)
        elements.append(Spacer(1, 20))
        
        # Facility information
        facility_info = [
            ["Facility:", self.metadata.facility_name],
            ["Report Date:", result.test_date.strftime("%B %d, %Y")],
            ["Report Time:", result.test_date.strftime("%H:%M:%S")],
            ["Software Version:", self.metadata.software_version],
            ["Certification:", self.metadata.certification]
        ]
        
        facility_table = Table(facility_info, colWidths=[2*inch, 4*inch])
        facility_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(facility_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_patient_info(self, result: DiagnosticResult) -> List:
        """Create patient information section"""
        
        elements = []
        
        # Section header
        header = Paragraph("Patient Information", self.header_style)
        elements.append(header)
        
        # Patient details
        patient_info = [
            ["Patient ID:", result.patient_id],
            ["Test Date:", result.test_date.strftime("%Y-%m-%d")],
            ["Analysis Type:", "EEG Symbolic Entropy Analysis"],
            ["Processing Status:", "Complete"]
        ]
        
        patient_table = Table(patient_info, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ]))
        
        elements.append(patient_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_diagnostic_summary(self, result: DiagnosticResult) -> List:
        """Create diagnostic summary section"""
        
        elements = []
        
        # Section header
        header = Paragraph("Diagnostic Summary", self.header_style)
        elements.append(header)
        
        # Primary finding
        confidence_text = f"{result.confidence:.1f}%"
        risk_color = self._get_risk_color(result.risk_level)
        
        summary_data = [
            ["Primary Finding:", result.condition],
            ["Confidence Level:", confidence_text],
            ["Risk Assessment:", result.risk_level],
            ["Entropy Signature:", f"{result.entropy_signature.get('primary', 0):.3f}"]
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
        summary_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightblue),
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 15))
        
        # Clinical interpretation
        interpretation = self._get_clinical_interpretation(result)
        interp_para = Paragraph(f"<b>Clinical Interpretation:</b> {interpretation}", 
                               self.finding_style)
        elements.append(interp_para)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_detailed_analysis(self, result: DiagnosticResult) -> List:
        """Create detailed analysis section"""
        
        elements = []
        
        # Section header
        header = Paragraph("Detailed Entropy Analysis", self.header_style)
        elements.append(header)
        
        # Entropy measurements
        entropy_data = [
            ["Measurement", "Value", "Reference Range", "Status"]
        ]
        
        for key, value in result.entropy_signature.items():
            ref_range = self._get_reference_range(key)
            status = self._get_measurement_status(key, value)
            entropy_data.append([
                key.replace('_', ' ').title(),
                f"{value:.3f}",
                ref_range,
                status
            ])
        
        entropy_table = Table(entropy_data, colWidths=[2*inch, 1*inch, 2*inch, 1*inch])
        entropy_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ]))
        
        elements.append(entropy_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_visualizations(self, result: DiagnosticResult) -> List:
        """Create visualization section"""
        
        elements = []
        
        # Section header
        header = Paragraph("Diagnostic Visualizations", self.header_style)
        elements.append(header)
        
        # Generate entropy signature plot
        plot_path = self._generate_entropy_plot(result)
        if plot_path:
            img = Image(plot_path, width=6*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 10))
        
        # Plot description
        plot_desc = Paragraph(
            "The entropy signature plot shows the patient's neural complexity "
            "patterns compared to healthy reference ranges. Deviations from "
            "normal ranges indicate potential neurological conditions.",
            self.finding_style
        )
        elements.append(plot_desc)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_recommendations(self, result: DiagnosticResult) -> List:
        """Create clinical recommendations section"""
        
        elements = []
        
        # Section header
        header = Paragraph("Clinical Recommendations", self.header_style)
        elements.append(header)
        
        # Recommendations list
        for i, recommendation in enumerate(result.recommendations, 1):
            rec_para = Paragraph(f"{i}. {recommendation}", self.finding_style)
            elements.append(rec_para)
        
        elements.append(Spacer(1, 15))
        
        # Disclaimer
        disclaimer = Paragraph(
            "<b>Important:</b> This analysis is for research purposes only and "
            "should not be used as the sole basis for clinical decisions. "
            "Please consult with a qualified neurologist for clinical interpretation.",
            self.warning_style
        )
        elements.append(disclaimer)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_technical_details(self, result: DiagnosticResult) -> List:
        """Create technical details section"""
        
        elements = []
        
        # Section header
        header = Paragraph("Technical Details", self.header_style)
        elements.append(header)
        
        # Technical parameters
        tech_data = []
        for key, value in result.technical_details.items():
            tech_data.append([key.replace('_', ' ').title(), str(value)])
        
        tech_table = Table(tech_data, colWidths=[3*inch, 3*inch])
        tech_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
        ]))
        
        elements.append(tech_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_footer(self) -> List:
        """Create report footer"""
        
        elements = []
        
        # Footer information
        footer_text = f"""
        Report generated by {self.metadata.software_version}<br/>
        Technician: {self.metadata.technician_name}<br/>
        Reviewing Physician: {self.metadata.physician_name}<br/>
        <br/>
        <b>Confidential Medical Information</b><br/>
        This report contains confidential patient information and should be handled 
        according to HIPAA privacy regulations.
        """
        
        footer_para = Paragraph(footer_text, self.styles['Normal'])
        elements.append(footer_para)
        
        return elements
    
    def _generate_entropy_plot(self, result: DiagnosticResult) -> Optional[str]:
        """Generate entropy signature visualization"""
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract entropy values
            measures = list(result.entropy_signature.keys())
            values = list(result.entropy_signature.values())
            
            # Create bar plot
            bars = ax.bar(measures, values, color='skyblue', alpha=0.7)
            
            # Add reference ranges (mock data for visualization)
            ref_ranges = [self._get_reference_range_values(measure) for measure in measures]
            
            for i, (bar, ref_range) in enumerate(zip(bars, ref_ranges)):
                if ref_range:
                    ax.axhline(y=ref_range[0], color='red', linestyle='--', alpha=0.5)
                    ax.axhline(y=ref_range[1], color='red', linestyle='--', alpha=0.5)
            
            ax.set_title('Entropy Signature Analysis', fontsize=14, fontweight='bold')
            ax.set_ylabel('Entropy Value')
            ax.set_xlabel('Measurement Type')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save plot
            plot_path = f"temp_entropy_plot_{result.patient_id}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            print(f"Could not generate plot: {e}")
            return None
    
    def _get_risk_color(self, risk_level: str) -> str:
        """Get color for risk level"""
        colors_map = {
            'LOW': 'green',
            'MODERATE': 'orange',
            'HIGH': 'red',
            'CRITICAL': 'darkred'
        }
        return colors_map.get(risk_level.upper(), 'black')
    
    def _get_clinical_interpretation(self, result: DiagnosticResult) -> str:
        """Get clinical interpretation text"""
        
        interpretations = {
            'Healthy': "Neural entropy patterns within normal ranges, indicating typical brain function.",
            'Epilepsy': "Reduced entropy patterns consistent with synchronized neural activity characteristic of epilepsy.",
            'Alcoholism': "Altered entropy signatures suggesting neural dysfunction associated with chronic alcohol use.",
            'Unknown': "Entropy patterns require further clinical correlation for definitive interpretation."
        }
        
        return interpretations.get(result.condition, interpretations['Unknown'])
    
    def _get_reference_range(self, measurement: str) -> str:
        """Get reference range for measurement"""
        
        ranges = {
            'symbolic_entropy': "3.5 - 4.0",
            'spectral_entropy': "10.5 - 11.5",
            'neural_flexibility': "0.02 - 0.08",
            'alpha_beta_ratio': "0.2 - 0.3"
        }
        
        return ranges.get(measurement, "N/A")
    
    def _get_reference_range_values(self, measurement: str) -> Optional[tuple]:
        """Get numeric reference range values"""
        
        ranges = {
            'symbolic_entropy': (3.5, 4.0),
            'spectral_entropy': (10.5, 11.5),
            'neural_flexibility': (0.02, 0.08),
            'alpha_beta_ratio': (0.2, 0.3)
        }
        
        return ranges.get(measurement)
    
    def _get_measurement_status(self, measurement: str, value: float) -> str:
        """Get status for measurement value"""
        
        ref_range = self._get_reference_range_values(measurement)
        if not ref_range:
            return "N/A"
        
        if ref_range[0] <= value <= ref_range[1]:
            return "Normal"
        elif value < ref_range[0]:
            return "Low"
        else:
            return "High"

def generate_sample_report():
    """Generate a sample clinical report for testing"""
    
    # Sample diagnostic result
    result = DiagnosticResult(
        patient_id="DEMO_001",
        test_date=datetime.now(),
        condition="Epilepsy",
        confidence=94.2,
        entropy_signature={
            'symbolic_entropy': 3.312,
            'spectral_entropy': 10.697,
            'neural_flexibility': 0.045,
            'alpha_beta_ratio': 0.228
        },
        risk_level="HIGH",
        recommendations=[
            "Recommend neurological consultation for epilepsy evaluation",
            "Consider EEG monitoring for seizure activity",
            "Review medication history for seizure-inducing substances",
            "Follow-up entropy analysis in 3-6 months"
        ],
        technical_details={
            'sampling_rate': '256 Hz',
            'signal_duration': '60 seconds',
            'processing_time': '2.3 seconds',
            'algorithm_version': 'CTEntropy v1.0',
            'confidence_interval': '89.1% - 97.8%'
        }
    )
    
    # Generate report
    reporter = ClinicalReporter()
    report_path = reporter.generate_diagnostic_report(result)
    
    print(f"Sample report generated: {report_path}")
    return report_path

if __name__ == "__main__":
    # Install required packages if not available
    try:
        from reportlab.lib.pagesizes import letter
    except ImportError:
        print("Installing reportlab for PDF generation...")
        import subprocess
        subprocess.check_call(["pip", "install", "reportlab"])
    
    generate_sample_report()