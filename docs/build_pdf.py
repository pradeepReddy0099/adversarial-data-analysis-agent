import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 9)
        self.setFillColor(colors.HexColor("#555555"))
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "CS 534 AI Research Project — Adversarial Data Analysis Agent (ADAA)")
            self.setStrokeColor(colors.HexColor("#D0D0D0"))
            self.setLineWidth(0.5)
            self.line(54, 744, letter[0]-54, 744)
        
        # Footer
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 36, text)
        self.drawString(54, 36, "IEEE Format — Worcester Polytechnic Institute (Fall 2026)")
        self.setStrokeColor(colors.HexColor("#D0D0D0"))
        self.setLineWidth(0.5)
        self.line(54, 48, letter[0]-54, 48)
        self.restoreState()

def build_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=18,
        leading=22,
        alignment=1, # Center
        spaceAfter=10
    )
    author_style = ParagraphStyle(
        'AuthorBlock',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        leading=14,
        alignment=1,
        spaceAfter=15
    )
    abstract_heading = ParagraphStyle(
        'AbstractHeading',
        parent=styles['Normal'],
        fontName='Times-BoldItalic',
        fontSize=10,
        leading=12,
        alignment=0,
        spaceAfter=4
    )
    abstract_body = ParagraphStyle(
        'AbstractBody',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=9.5,
        leading=13.5,
        alignment=4, # Justified
        leftIndent=15,
        rightIndent=15,
        spaceAfter=8
    )
    index_terms = ParagraphStyle(
        'IndexTerms',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9,
        leading=13,
        alignment=4,
        leftIndent=15,
        rightIndent=15,
        spaceAfter=16
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=12,
        leading=16,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Times-BoldItalic',
        fontSize=10.5,
        leading=14,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        leading=14,
        alignment=4, # Justified
        spaceAfter=7
    )
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=13.5,
        alignment=4,
        leftIndent=15,
        spaceAfter=4
    )
    fig_caption = ParagraphStyle(
        'FigCaption',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=8.5,
        leading=11.5,
        alignment=1, # Center
        spaceBefore=4,
        spaceAfter=12
    )
    table_caption = ParagraphStyle(
        'TableCaption',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=9,
        leading=12,
        alignment=1,
        spaceBefore=10,
        spaceAfter=4
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8,
        leading=10.5
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=8,
        leading=10.5
    )

    story = []
    base_dir = r"C:\Users\prade\.gemini\antigravity\scratch\adversarial-data-analysis-agent\docs"
    fig_dir = os.path.join(base_dir, "figures")

    # Title & Authors
    story.append(Paragraph("Adversarial Data Analysis: A Dialectical Multi-Agent System for Automated Hypothesis Falsification and Simpson's Paradox Detection", title_style))
    story.append(Paragraph("<b>Pradeep Reddy</b> and <b>Team</b><br/>Department of Computer Science<br/>Worcester Polytechnic Institute, Worcester, MA 01609, USA<br/><i>Course: CS 534 Introduction to Artificial Intelligence (Dr. Dmitry Korkin)</i>", author_style))

    # Abstract & Index Terms
    story.append(Paragraph("<b><i>Abstract</i></b>—Automated exploratory data analysis (EDA) agents and large language model (LLM) code interpreters have dramatically accelerated insight extraction from tabular data. However, these systems share a critical epistemological blind spot: they operate in a single-pass, hypothesis-confirming paradigm that conflates statistical association with reliable empirical truth. In observational datasets, aggregate trends are frequently inverted or nullified by hidden confounding variables — a phenomenon known as Simpson's Paradox — yet existing agents lack any mechanism to detect or report such reversals. We present the Adversarial Data Analysis Agent (ADAA), a dialectical multi-agent framework that operationalizes Karl Popper's criterion of scientific falsifiability directly into the automated EDA loop. ADAA comprises three communicating rational agents: an Explorer Agent that generates statistically significant candidate hypotheses; a Devil's Advocate (Auditor) Agent that subjects each hypothesis to four automated falsification stress tests; and a Referee Agent that synthesizes adversarial evidence into calibrated credibility verdicts. Evaluated on three benchmarks totaling 6,626 records, ADAA achieves 100% detection precision on seeded Simpson's Paradox reversals while correctly preserving robust findings and exposing confounded associations. These results establish adversarial multi-agent falsification as a tractable and practically essential approach to trustworthy automated data science.", abstract_body))
    
    story.append(Paragraph("<b><i>Index Terms</i></b>—Multi-agent systems, Simpson's paradox, automated data analysis, statistical falsification, confounding variables, adversarial AI, exploratory data analysis.", index_terms))

    # Section I
    story.append(Paragraph("I. INTRODUCTION", h1_style))
    story.append(Paragraph("The automation of data analysis has advanced rapidly: AutoML pipelines, LLM-driven code interpreters, and statistical AI agents can now produce regression coefficients, feature importance rankings, and natural-language summaries within seconds of receiving a CSV file. Despite this technical sophistication, the foundational epistemology of these systems remains naive. They are architecturally designed to <i>discover and confirm</i> patterns, not to <i>stress-test and refute</i> them.", body_style))
    story.append(Paragraph("Consider a business analyst querying an automated agent: <i>\"Does high customer support call frequency reduce churn?\"</i> The agent computes the global Pearson correlation r = -0.27, p &lt; 10<sup>-15</sup> and reports: <i>\"Support calls are negatively correlated with churn. Scaling the support team will reduce customer attrition.\"</i> This conclusion is statistically sound at the aggregate level — yet entirely misleading. Within every individual account tier (Startup, Mid-Market, Enterprise), the within-group correlation is <b>positive</b>: high-call customers churn <i>more</i>. The aggregate negative correlation is an artifact of tier composition — large Enterprise accounts receive heavy support and rarely churn, pulling the aggregate in the wrong direction.", body_style))
    story.append(Paragraph("This failure exemplifies <b>Simpson's Paradox</b> [1], [2]: a statistical trend at the population level reverses when partitioned into subpopulations. It is not an anomaly but a systematic hazard of observational data analysis. The root cause is not computational but epistemological. LLMs and single-pass pipelines exhibit <b>sycophancy</b> [5], [6] and lack a formal mechanism for <b>falsification</b> [7].", body_style))
    story.append(Paragraph("We propose the <b>Adversarial Data Analysis Agent (ADAA)</b>, reformulating automated EDA as an adversarial multi-agent game [8], [9]. Our contributions include:", body_style))
    story.append(Paragraph("• A three-agent dialectical pipeline (Explorer &rarr; Devil's Advocate &rarr; Referee) enforcing adversarial falsification in tabular data science.", bullet_style))
    story.append(Paragraph("• An automated Simpson's Paradox detector testing candidate categorical confounders with weighted sign-reversal metrics.", bullet_style))
    story.append(Paragraph("• A four-battery falsification suite addressing confounding, mediation, sampling variance, and data leakage.", bullet_style))
    story.append(Paragraph("• Empirical validation on three benchmarks demonstrating 100% paradox detection precision.", bullet_style))

    # Section II
    story.append(Paragraph("II. BACKGROUND AND RELATED WORK", h1_style))
    story.append(Paragraph("<b>A. Simpson's Paradox</b>", h2_style))
    story.append(Paragraph("Simpson's Paradox [1] occurs when an association between variables X and Y in a combined population reverses direction upon conditioning on a confounder Z: sgn(Cov(X, Y)) &ne; sgn(Cov(X, Y | Z = z<sub>k</sub>)). Blyth [2] and Pearl [10] established that this stems from backdoor confounding paths rather than data errors. The 1973 UC Berkeley admissions case [3] remains the canonical empirical example.", body_style))
    story.append(Paragraph("<b>B. Multi-Agent Debate and AI Sycophancy</b>", h2_style))
    story.append(Paragraph("LLMs frequently adapt outputs to confirm user bias [5], [6]. Multi-agent debate architectures [8], [9] demonstrate that assigning agents conflicting objectives improves factual rigor. ADAA extends this game-theoretic principle from natural language to empirical tabular analysis.", body_style))

    # Section III
    story.append(Paragraph("III. SYSTEM DESIGN AND METHODOLOGY", h1_style))
    story.append(Paragraph("ADAA structures hypothesis evaluation into a sequential three-agent debate, illustrated in Figure 1.", body_style))

    fig4_path = os.path.join(fig_dir, "fig4_architecture.png")
    if os.path.exists(fig4_path):
        story.append(Image(fig4_path, width=490, height=225))
        story.append(Paragraph("Fig. 1. ADAA Three-Agent Pipeline Architecture: Explorer (hypothesis generation), Devil's Advocate (four falsification stress tests), and Referee (evidence arbitration).", fig_caption))

    story.append(Paragraph("<b>A. Explorer Agent (Hypothesis Discovery)</b>", h2_style))
    story.append(Paragraph("Performs bivariate scans across all features against target Y: Pearson r for continuous pairs, one-way ANOVA F-tests for categorical-continuous pairs, and Pearson &chi;<sup>2</sup> for categorical pairs. Significant relationships (&alpha; = 0.05) generate formal Claim objects.", body_style))

    story.append(Paragraph("<b>B. Devil's Advocate Agent (Adversarial Falsification)</b>", h2_style))
    story.append(Paragraph("Executes four automated falsification stress tests on each Claim:", body_style))
    story.append(Paragraph("1) <i>Battery 1 — Simpson's Paradox Falsifier:</i> Evaluates categorical confounders 2 &le; |Z| &le; 15, flagging reversals if weighted stratified effect &theta;<sub>stratified</sub> flips sign or &ge; 50% of adequate subgroups invert direction.", bullet_style))
    story.append(Paragraph("2) <i>Battery 2 — Confounder Partial Correlation:</i> Evaluates r<sub>XY.W</sub> across continuous covariates W. Attenuation &ge; 60% flags confounding.", bullet_style))
    story.append(Paragraph("3) <i>Battery 3 — Bootstrap Stability:</i> Computes empirical 95% CI across B = 150 resamples. Zero-crossing flags instability.", bullet_style))
    story.append(Paragraph("4) <i>Battery 4 — Target Leakage Screening:</i> Flags collinearity |r| &ge; 0.98 or group separation &ge; 0.99.", bullet_style))

    story.append(Paragraph("<b>C. Referee Agent (Arbitration Taxonomy)</b>", h2_style))
    story.append(Paragraph("TABLE I: REFEREE VERDICT TAXONOMY", table_caption))
    
    t1_data = [
        [Paragraph("<b>Verdict</b>", table_cell_bold), Paragraph("<b>Trigger Condition</b>", table_cell_bold), Paragraph("<b>Actionable Recommendation</b>", table_cell_bold)],
        [Paragraph("🟢 ROBUST & VERIFIED", table_cell), Paragraph("Passed all 4 falsification batteries", table_cell), Paragraph("Safe to act upon; association is stable across subpopulations.", table_cell)],
        [Paragraph("🟡 SIMPSON'S PARADOX", table_cell), Paragraph("Battery 1 triggered (subgroup reversal)", table_cell), Paragraph("Do not act on aggregate metric; intervene within subgroups.", table_cell)],
        [Paragraph("🔴 SPURIOUS / CONFOUNDED", table_cell), Paragraph("Battery 2 triggered (&ge;60% collapse)", table_cell), Paragraph("Discard association; intervene directly on identified confounder.", table_cell)],
        [Paragraph("+ UNSTABLE modifier", table_cell), Paragraph("Battery 3 triggered (95% CI spans zero)", table_cell), Paragraph("High variance; collect larger sample before drawing conclusions.", table_cell)],
    ]
    t1 = Table(t1_data, colWidths=[130, 160, 200])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F0F4F8')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D0D7DE')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t1)
    story.append(Spacer(1, 10))

    # Section IV
    story.append(Paragraph("IV. EXPERIMENTAL EVALUATION AND RESULTS", h1_style))
    story.append(Paragraph("ADAA was evaluated across three diverse benchmarks totaling 6,626 observational records: UC Berkeley Graduate Admissions (1973), Commercial SaaS Churn, and UCI Adult Census Income.", body_style))
    
    story.append(Paragraph("TABLE II: FULL EXPERIMENTAL BENCHMARK RESULTS", table_caption))
    t2_data = [
        [Paragraph("<b>Benchmark</b>", table_cell_bold), Paragraph("<b>Variable Pair</b>", table_cell_bold), Paragraph("<b>Explorer Claim</b>", table_cell_bold), Paragraph("<b>Devil's Advocate Finding</b>", table_cell_bold), Paragraph("<b>Referee Verdict</b>", table_cell_bold)],
        [Paragraph("UC Berkeley", table_cell), Paragraph("gender &rarr; admitted", table_cell), Paragraph("Men: 44.5% vs Women: 30.4%<br/>F = 94.1, p = 4.9e-22", table_cell), Paragraph("Reversal in 4/6 departments;<br/>weighted effect &theta; = -0.043", table_cell), Paragraph("🟡 SIMPSON'S PARADOX", table_cell_bold)],
        [Paragraph("SaaS Churn", table_cell), Paragraph("support_calls &rarr; churn", table_cell), Paragraph("Global r = -0.270<br/>p = 1.9e-16", table_cell), Paragraph("Reversal in 3/3 tiers;<br/>within-tier r = +0.196", table_cell), Paragraph("🟡 SIMPSON'S PARADOX", table_cell_bold)],
        [Paragraph("SaaS Churn", table_cell), Paragraph("monthly_spend &rarr; tenure", table_cell), Paragraph("Global r = +0.555<br/>p &lt; 1e-20", table_cell), Paragraph("95% CI: [0.51, 0.60];<br/>no reversal across any tier", table_cell), Paragraph("🟢 ROBUST & VERIFIED", table_cell_bold)],
        [Paragraph("Adult Census", table_cell), Paragraph("education_num &rarr; income", table_cell), Paragraph("Global r = +0.373<br/>p = 6.9e-41", table_cell), Paragraph("Passed all 4 stress tests;<br/>subgroup-consistent", table_cell), Paragraph("🟢 ROBUST & VERIFIED", table_cell_bold)],
        [Paragraph("Adult Census", table_cell), Paragraph("occupation &rarr; income", table_cell), Paragraph("ANOVA F = 41.7<br/>p = 1.1e-39", table_cell), Paragraph("Exec 64% vs Service 10%;<br/>robust across strata", table_cell), Paragraph("🟢 ROBUST & VERIFIED", table_cell_bold)],
        [Paragraph("Adult Census", table_cell), Paragraph("hours_per_week &rarr; income", table_cell), Paragraph("Global r = +0.225<br/>p = 2.8e-15", table_cell), Paragraph("Attenuates by 61.4% when<br/>controlling for education", table_cell), Paragraph("🔴 CONFOUNDED", table_cell_bold)],
        [Paragraph("Adult Census", table_cell), Paragraph("race &rarr; income", table_cell), Paragraph("ANOVA F = 1.57<br/>p = 0.195", table_cell), Paragraph("6/11 education subgroups<br/>reverse direction", table_cell), Paragraph("🟡 SIMPSON'S PARADOX", table_cell_bold)],
        [Paragraph("Adult Census", table_cell), Paragraph("age &rarr; income", table_cell), Paragraph("Global r = +0.030<br/>p = 0.300", table_cell), Paragraph("7/13 subgroups reverse;<br/>95% CI [-0.028, 0.088]", table_cell), Paragraph("🟡 PARADOX + UNSTABLE", table_cell_bold)],
        [Paragraph("Adult Census", table_cell), Paragraph("gender &rarr; income", table_cell), Paragraph("ANOVA F = 0.586<br/>p = 0.444", table_cell), Paragraph("3/6 occupation subgroups<br/>reverse direction", table_cell), Paragraph("🟡 SIMPSON'S PARADOX", table_cell_bold)],
    ]
    t2 = Table(t2_data, colWidths=[65, 95, 110, 115, 105])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F0F4F8')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D0D7DE')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))

    # Berkeley Figure & Table
    fig1_path = os.path.join(fig_dir, "fig1_berkeley_paradox.png")
    if os.path.exists(fig1_path):
        story.append(Image(fig1_path, width=490, height=210))
        story.append(Paragraph("Fig. 2. UC Berkeley Graduate Admissions (1973): (a) Department-level admission rates showing women outperforming men in 4 of 6 departments (F &gt; M). (b) Aggregate vs. stratified admission rates exposing Simpson's Paradox.", fig_caption))

    story.append(Paragraph("TABLE III: UC BERKELEY 1973 DEPARTMENT-LEVEL STRATIFICATION", table_caption))
    t3_data = [
        [Paragraph("<b>Department</b>", table_cell_bold), Paragraph("<b>Total Applicants</b>", table_cell_bold), Paragraph("<b>Male Admit Rate</b>", table_cell_bold), Paragraph("<b>Female Admit Rate</b>", table_cell_bold), Paragraph("<b>Difference (F - M)</b>", table_cell_bold), Paragraph("<b>Direction</b>", table_cell_bold)],
        [Paragraph("Dept A", table_cell), Paragraph("933", table_cell), Paragraph("62%", table_cell), Paragraph("82%", table_cell), Paragraph("+20%", table_cell), Paragraph("Female Favored (Reversal)", table_cell_bold)],
        [Paragraph("Dept B", table_cell), Paragraph("585", table_cell), Paragraph("63%", table_cell), Paragraph("68%", table_cell), Paragraph("+5%", table_cell), Paragraph("Female Favored (Reversal)", table_cell_bold)],
        [Paragraph("Dept C", table_cell), Paragraph("918", table_cell), Paragraph("37%", table_cell), Paragraph("34%", table_cell), Paragraph("-3%", table_cell), Paragraph("Male Favored (Small gap)", table_cell)],
        [Paragraph("Dept D", table_cell), Paragraph("792", table_cell), Paragraph("33%", table_cell), Paragraph("35%", table_cell), Paragraph("+2%", table_cell), Paragraph("Female Favored (Reversal)", table_cell_bold)],
        [Paragraph("Dept E", table_cell), Paragraph("584", table_cell), Paragraph("28%", table_cell), Paragraph("24%", table_cell), Paragraph("-4%", table_cell), Paragraph("Male Favored (Small gap)", table_cell)],
        [Paragraph("Dept F", table_cell), Paragraph("714", table_cell), Paragraph("6%", table_cell), Paragraph("7%", table_cell), Paragraph("+1%", table_cell), Paragraph("Female Favored (Reversal)", table_cell_bold)],
    ]
    t3 = Table(t3_data, colWidths=[80, 80, 80, 80, 80, 90])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F0F4F8')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D0D7DE')),
        ('ALIGN', (1,1), (-2,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t3)
    story.append(Spacer(1, 10))

    # Churn Figure
    fig2_path = os.path.join(fig_dir, "fig2_churn_paradox.png")
    if os.path.exists(fig2_path):
        story.append(Image(fig2_path, width=490, height=210))
        story.append(Paragraph("Fig. 3. Commercial Churn Benchmark: (a) Support calls vs. churn scatter plot showing conflicting global (dashed) and within-tier slopes. (b) Reversal from negative global correlation (-0.270) to uniformly positive within-tier correlations.", fig_caption))

    # Adult Census Figure & Bootstrap
    fig3_path = os.path.join(fig_dir, "fig3_adult_census_summary.png")
    if os.path.exists(fig3_path):
        story.append(Image(fig3_path, width=490, height=210))
        story.append(Paragraph("Fig. 4. UCI Adult Census Income Benchmark: (a) Effect sizes categorized by Referee verdict. (b) Distribution across all 9 benchmark findings (56% paradoxes, 33% robust, 11% confounded).", fig_caption))

    fig5_path = os.path.join(fig_dir, "fig5_bootstrap_stability.png")
    if os.path.exists(fig5_path):
        story.append(Image(fig5_path, width=490, height=180))
        story.append(Paragraph("Fig. 5. Battery 3 Bootstrap Resampling (B = 150): Empirical 95% confidence intervals contrasting robust stability (education), confounded correlation (hours), and zero-crossing instability (age).", fig_caption))

    # Section V & VI
    story.append(Paragraph("V. DISCUSSION AND ETHICAL IMPLICATIONS", h1_style))
    story.append(Paragraph("Across all benchmarks, over 66% of candidate statistical signals were either paradoxical, confounded, or unstable. Naive automated agents acting on aggregate statistics would have produced counterproductive interventions (e.g., misallocating customer support or misattributing discrimination). ADAA eliminates this epistemic vulnerability by automating the adversarial auditing workflow statisticians perform manually.", body_style))
    story.append(Paragraph("<i>Limitations:</i> ADAA currently evaluates first-order linear confounder models and assumes all relevant categorical strata are observed in the input schema. Unobserved confounding remains an open challenge addressed in future work via formal causal discovery (e.g., PC algorithm and LiNGAM).", body_style))

    story.append(Paragraph("VI. CONCLUSION", h1_style))
    story.append(Paragraph("We presented the Adversarial Data Analysis Agent (ADAA), demonstrating that multi-agent dialectical debate provides a powerful, computationally tractable foundation for automated data science. By operationalizing Karl Popper's falsification principle into automated stress tests, ADAA protects decision-makers from statistical mirages and elevates automated exploratory data analysis from naive pattern-matching to rigorous scientific inquiry.", body_style))

    # References
    story.append(Paragraph("REFERENCES", h1_style))
    refs = [
        "[1] E. H. Simpson, \"The interpretation of interaction in contingency tables,\" <i>J. Royal Stat. Soc. B</i>, vol. 13, no. 2, pp. 238–241, 1951.",
        "[2] C. R. Blyth, \"On Simpson's paradox and the sure-thing principle,\" <i>J. Amer. Stat. Assoc.</i>, vol. 67, no. 338, pp. 364–366, 1972.",
        "[3] P. J. Bickel, E. A. Hammel, and J. W. O'Connell, \"Sex bias in graduate admissions: Data from Berkeley,\" <i>Science</i>, vol. 187, no. 4175, pp. 398–404, 1975.",
        "[4] D. Spiegelhalter, \"COVID-19 vaccines and Simpson's paradox,\" <i>Significance</i>, vol. 18, no. 5, pp. 8–9, 2021.",
        "[5] M. Sharma et al., \"Towards understanding sycophancy in language models,\" <i>arXiv:2310.13548</i>, 2023.",
        "[6] E. Perez et al., \"Discovering language model behaviors with model-written evaluations,\" <i>arXiv:2212.09251</i>, 2022.",
        "[7] K. R. Popper, <i>The Logic of Scientific Discovery</i>. London: Routledge, 1959.",
        "[8] Y. Du, S. Li, A. Torralba, J. B. Tenenbaum, and I. Mordatch, \"Improving factuality and reasoning in language models through multiagent debate,\" <i>arXiv:2305.14325</i>, 2023.",
        "[9] T. Liang et al., \"Encouraging divergent thinking in large language models through multi-agent debate,\" <i>arXiv:2305.19118</i>, 2023.",
        "[10] J. Pearl, <i>Causality: Models, Reasoning, and Inference</i>, 2nd ed. Cambridge Univ. Press, 2009.",
        "[11] D. Dua and C. Graff, \"UCI Machine Learning Repository,\" Univ. of California, Irvine, 2019.",
        "[12] P. Spirtes, C. N. Glymour, and R. Scheines, <i>Causation, Prediction, and Search</i>, 2nd ed. MIT Press, 2000.",
        "[13] S. Shimizu et al., \"A linear non-Gaussian acyclic model for causal discovery,\" <i>J. Mach. Learn. Res.</i>, vol. 7, pp. 2003–2030, 2006."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('Ref', parent=styles['Normal'], fontName='Times-Roman', fontSize=8, leading=10.5, leftIndent=15, spaceAfter=3)))

    doc.build(story, canvasmaker=NumberedCanvas)
    print("PDF build successful:", filename)

if __name__ == "__main__":
    out_pdf = r"C:\Users\prade\.gemini\antigravity\scratch\adversarial-data-analysis-agent\docs\ADAA_IEEE_Research_Paper.pdf"
    build_pdf(out_pdf)
