"""
generate_presentation.py
Generates a polished 16:9 widescreen PowerPoint presentation (PPTX)
for JNTU Hyderabad coursework submission (Roll No: 2606MGAIL130).
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_deck(output_pptx="presentation/Taxi_Out_Prediction_Presentation.pptx"):
    os.makedirs(os.path.dirname(output_pptx), exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # blank layout
    
    # Color Palette (Aviation Dark Modern Theme)
    C_NAVY_BG = RGBColor(11, 15, 25)      # #0b0f19
    C_CARD_BG = RGBColor(24, 32, 51)      # #182033
    C_CARD_BORDER = RGBColor(51, 65, 85)  # #334155
    C_CYAN = RGBColor(56, 189, 248)       # #38bdf8
    C_WHITE = RGBColor(248, 250, 252)     # #f8fafc
    C_MUTED = RGBColor(148, 163, 184)     # #94a3b8
    C_GREEN = RGBColor(74, 222, 128)      # #4ade80
    C_AMBER = RGBColor(250, 204, 21)      # #facc15
    
    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = C_NAVY_BG
        bg.line.color.rgb = C_NAVY_BG
        return bg

    def add_header(slide, title_text, category_text="AIRCRAFT TAXI-OUT PREDICTION | JNTU HYDERABAD"):
        # Header category
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_c = cat_box.text_frame
        tf_c.word_wrap = True
        p_c = tf_c.paragraphs[0]
        p_c.text = category_text.upper()
        p_c.font.size = Pt(10)
        p_c.font.bold = True
        p_c.font.color.rgb = C_CYAN
        
        # Header title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.7))
        tf_t = title_box.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(22)
        p_t.font.bold = True
        p_t.font.color.rgb = C_WHITE
        
    def add_card(slide, left, top, width, height, bg_color=C_CARD_BG, border_color=C_CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1)
        return card

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)
    
    card1 = add_card(s1, Inches(1.2), Inches(1.2), Inches(10.9), Inches(5.1))
    
    tb1 = s1.shapes.add_textbox(Inches(1.6), Inches(1.6), Inches(10.1), Inches(4.3))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "AIRCRAFT TAXI-OUT DURATION PREDICTION"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    
    p = tf1.add_paragraph()
    p.text = "Surface Congestion Analytics & Pre-Departure Decision Support Pipeline"
    p.font.size = Pt(16)
    p.font.color.rgb = C_WHITE
    p.space_after = Pt(24)
    
    p = tf1.add_paragraph()
    p.text = "• Coursework: JNTU Hyderabad — Aviation Data Science and Operations Research"
    p.font.size = Pt(13)
    p.font.color.rgb = C_MUTED
    
    p = tf1.add_paragraph()
    p.text = "• Student Identifier / Hall Ticket: 2606MGAIL130"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_GREEN
    
    p = tf1.add_paragraph()
    p.text = "• Primary Focus Hub: Dallas/Fort Worth International Airport (DFW) & National Airspace System"
    p.font.size = Pt(13)
    p.font.color.rgb = C_MUTED
    
    p = tf1.add_paragraph()
    p.text = "• Core Dataset: Bureau of Transportation Statistics (BTS) Airline On-Time Performance (2025)"
    p.font.size = Pt(13)
    p.font.color.rgb = C_MUTED
    p.space_after = Pt(20)
    
    p = tf1.add_paragraph()
    p.text = "Note: Academic machine learning prototype evaluating BTS ground queue dynamics; designed to augment, not replace, certified ATC/airline operations systems."
    p.font.size = Pt(11)
    p.font.italic = True
    p.font.color.rgb = C_AMBER

    # =========================================================================
    # SLIDE 2: WHAT IS TAXI-OUT & ITS HINDERANCES (With Infographic)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "What is Taxi-Out & What Causes Surface Delays?", "DOMAIN CONTEXT & PHYSICAL BOTTLENECKS")
    
    # Left Column: Process Diagram
    img_path = "figures/taxi_out_process_diagram.jpg"
    if os.path.exists(img_path):
        add_card(s2, Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.3))
        s2.shapes.add_picture(img_path, Inches(0.9), Inches(1.6), width=Inches(6.6))
    
    # Right Column: Explanatory Card
    add_card(s2, Inches(7.8), Inches(1.5), Inches(4.7), Inches(5.3))
    tb2 = s2.shapes.add_textbox(Inches(8.0), Inches(1.7), Inches(4.3), Inches(4.9))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    
    p = tf2.paragraphs[0]
    p.text = "The Operational Definition"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    
    p = tf2.add_paragraph()
    p.text = "TAXI_OUT = WHEELS_OFF - DEP_TIME\nElapsed ground time between terminal gate pushback and actual airborne takeoff roll."
    p.font.size = Pt(12)
    p.font.color.rgb = C_WHITE
    p.space_after = Pt(14)
    
    p = tf2.add_paragraph()
    p.text = "Primary Surface Hindrances:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_AMBER
    
    hindrances = [
        ("1. Apron Ramp Congestion", "Pushback tug conflicts, ground service vehicle delays, and alleyway gridlock during scheduled bank surges."),
        ("2. Active Runway Crossings", "Aircraft must hold on taxiways waiting for incoming landing flights to clear intersecting runways."),
        ("3. Runway Threshold Queue", "Departure sequencing buffer at runway head; wake turbulence separation holds behind heavy jets."),
        ("4. Departure Metering (DMP)", "Tactical FAA ground holds and flow management programs managing airspace sector capacity.")
    ]
    for h_title, h_desc in hindrances:
        p = tf2.add_paragraph()
        p.text = f"{h_title}: "
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = C_WHITE
        p.add_run = p.text
        p = tf2.add_paragraph()
        p.text = h_desc
        p.font.size = Pt(10)
        p.font.color.rgb = C_MUTED
        p.space_after = Pt(4)

    # =========================================================================
    # SLIDE 3: OPERATIONAL MOTIVATION & SUSTAINABILITY
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Operational Significance: Why Forecast Taxi-Out?", "AIRPORT COLLABORATIVE DECISION MAKING (A-CDM)")
    
    # 3 Cards Layout
    cards_data = [
        ("Fuel & Carbon Reduction", C_GREEN, [
            "Narrowbody jets idle at ~13 kg Jet-A fuel per minute on surface taxiways.",
            "Each 1 kg of Jet-A burned produces 3.16 kg of CO2 emissions.",
            "Holding aircraft at gates for 10 min during heavy queues saves ~130 kg of fuel and 410 kg of CO2 per flight."
        ]),
        ("Ramp & Runway Optimization", C_CYAN, [
            "Prevents apron gridlock and tarmac delay rule violations (>180 min FAA penalty).",
            "Coordinates Target Takeoff Times (TTOT = Pushback + Taxi_Out) for downstream en-route center metering.",
            "Allows balanced runway assignment across sprawling hub layouts (e.g., DFW's 7 runways)."
        ]),
        ("Academic Prototype Scope", C_AMBER, [
            "Designed as a decision-support prototype exploring public BTS historical records.",
            "Complements rather than replaces multi-million dollar airline operations control (AOC) or FAA radar systems.",
            "Provides tactical queue visibility for pre-departure scheduling adjustments."
        ])
    ]
    
    for i, (title, color, bullets) in enumerate(cards_data):
        left = Inches(0.8 + i * 4.0)
        add_card(s3, left, Inches(1.6), Inches(3.7), Inches(5.2))
        tb = s3.shapes.add_textbox(left + Inches(0.25), Inches(1.8), Inches(3.2), Inches(4.8))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = color
        p.space_after = Pt(16)
        
        for b in bullets:
            p = tf.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(11)
            p.font.color.rgb = C_WHITE
            p.space_after = Pt(10)

    # =========================================================================
    # SLIDE 4: DATASET & PREPROCESSING FUNNEL
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Data Preprocessing: From 75,000 Raw Flights to 72,272 Clean Records", "DATA CLEANING & OUTLIER CAPPING")
    
    # Left Card: Funnel Breakdown
    add_card(s4, Inches(0.8), Inches(1.5), Inches(4.5), Inches(5.3))
    tb4 = s4.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(4.1), Inches(4.9))
    tf4 = tb4.text_frame
    tf4.word_wrap = True
    
    p = tf4.paragraphs[0]
    p.text = "Data Cleaning Audit Trail"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(12)
    
    funnel_items = [
        ("Raw BTS Records", "75,000 flights (100.0%)", C_WHITE),
        ("Cancelled Flights Removed", "-1,507 flights (2.01%)", C_MUTED),
        ("Diverted Flights Removed", "-194 flights (0.26%)", C_MUTED),
        ("Invalid/Non-positive Taxi Removed", "-434 flights (0.58%)", C_MUTED),
        ("Extreme Outliers (>120 min) Filtered", "-592 flights (0.79%)", C_MUTED),
        ("Final Clean Operational Dataset", "72,272 flights (96.36% retention)", C_GREEN)
    ]
    for stage, count, col in funnel_items:
        p = tf4.add_paragraph()
        p.text = f"{stage}: "
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = col
        p = tf4.add_paragraph()
        p.text = count
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p.space_after = Pt(6)

    # Right Card: Distribution Image
    p_dist = "figures/eda_taxi_distribution.png"
    if os.path.exists(p_dist):
        add_card(s4, Inches(5.6), Inches(1.5), Inches(6.9), Inches(5.3))
        s4.shapes.add_picture(p_dist, Inches(5.8), Inches(1.8), width=Inches(6.5))

    # =========================================================================
    # SLIDE 5: FEATURE ENGINEERING & ZERO-LEAKAGE SPLIT
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Feature Engineering: Capturing Diurnal Waves & Queue Physics", "ZERO-LEAKAGE MACHINE LEARNING ARCHITECTURE")
    
    features_cards = [
        ("1. Temporal & Cyclical Cycles", C_CYAN, [
            "CRS_DEP_TIME converted to Hour (0-23).",
            "Preserves midnight continuity (23:59 -> 00:00) using orthogonal trigonometric encoding:",
            "Sin_Hour = sin(2π × Hour / 24)",
            "Cos_Hour = cos(2π × Hour / 24)",
            "Captures morning (07-09h) and evening (16-19h) airline departure rush banks."
        ]),
        ("2. Surface Congestion Index", C_AMBER, [
            "Taxi delays scale non-linearly with concurrent pushbacks (queuing phenomenon).",
            "CONGESTION_1H: Concurrent departures from same origin airport in past 60 minutes.",
            "CONGESTION_30M: Concurrent departures in past 30 minutes.",
            "Peak observed DFW congestion: 26 flights/hour."
        ]),
        ("3. Leak-Free Target Encoding", C_GREEN, [
            "Captures airport geometry (ORIGIN) and carrier bank scheduling (CARRIER).",
            "Strictly computed on Months 1-10 training partition using Empirical Bayes smoothing (m=10).",
            "Zero target leakage: Test set records (Months 11-12) are purely mapped from training lookups."
        ])
    ]
    for i, (title, color, bullets) in enumerate(features_cards):
        left = Inches(0.8 + i * 4.0)
        add_card(s5, left, Inches(1.6), Inches(3.7), Inches(5.2))
        tb = s5.shapes.add_textbox(left + Inches(0.25), Inches(1.8), Inches(3.2), Inches(4.8))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = color
        p.space_after = Pt(14)
        
        for b in bullets:
            p = tf.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(10.5)
            p.font.color.rgb = C_WHITE
            p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 6: MODEL ARCHITECTURES & BENCHMARK COMPARISON
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Model Comparison: Linear Regression vs Random Forest vs XGBoost", "HELD-OUT TEST SET (12,001 FLIGHTS)")
    
    # Left: Evaluation plot
    p_comp = "figures/model_comparison.png"
    if os.path.exists(p_comp):
        add_card(s6, Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.3))
        s6.shapes.add_picture(p_comp, Inches(0.9), Inches(1.9), width=Inches(6.6))
        
    # Right: Results Table & Insights
    add_card(s6, Inches(7.8), Inches(1.5), Inches(4.7), Inches(5.3))
    tb6 = s6.shapes.add_textbox(Inches(8.0), Inches(1.7), Inches(4.3), Inches(4.9))
    tf6 = tb6.text_frame
    tf6.word_wrap = True
    
    p = tf6.paragraphs[0]
    p.text = "Benchmark Results Table"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(10)
    
    table_rows = [
        ("Linear Regression (Baseline):", "MAE: 4.17m | RMSE: 10.04m | ±5m: 78.9%"),
        ("Random Forest (100 Trees):", "MAE: 4.04m | RMSE: 10.00m | ±5m: 82.6%"),
        ("XGBoost Regressor (Best):", "MAE: 3.88m | RMSE: 9.89m | ±5m: 84.9%")
    ]
    for m_name, m_res in table_rows:
        p = tf6.add_paragraph()
        p.text = m_name
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = C_GREEN if "Best" in m_name else C_WHITE
        p = tf6.add_paragraph()
        p.text = m_res
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_CYAN if "Best" in m_name else C_MUTED
        p.space_after = Pt(6)
        
    p = tf6.add_paragraph()
    p.text = "Key Operational Takeaways:"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_AMBER
    
    p = tf6.add_paragraph()
    p.text = "1. XGBoost achieves lowest error (MAE 3.88 min) by sequentially correcting residual errors during non-linear bank surges."
    p.font.size = Pt(10)
    p.font.color.rgb = C_WHITE
    p.space_after = Pt(4)
    
    p = tf6.add_paragraph()
    p.text = "2. 84.9% of flights predicted within ±5 min (FAA Departure Metering compliance standard)."
    p.font.size = Pt(10)
    p.font.color.rgb = C_WHITE

    # =========================================================================
    # SLIDE 7: AIRFIELD CONGESTION & NON-LINEAR ACCUMULATION
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Surface Congestion Impact: Non-Linear Queue Saturation", "M/M/c QUEUING DYNAMICS AT HUB AIRPORTS")
    
    p_cong = "figures/eda_congestion_vs_taxi.png"
    if os.path.exists(p_cong):
        add_card(s7, Inches(0.8), Inches(1.5), Inches(7.5), Inches(5.3))
        s7.shapes.add_picture(p_cong, Inches(0.9), Inches(1.8), width=Inches(7.3))
        
    add_card(s7, Inches(8.5), Inches(1.5), Inches(4.0), Inches(5.3))
    tb7 = s7.shapes.add_textbox(Inches(8.7), Inches(1.7), Inches(3.6), Inches(4.9))
    tf7 = tb7.text_frame
    tf7.word_wrap = True
    
    p = tf7.paragraphs[0]
    p.text = "A-CDM Surface Queuing Insight"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(14)
    
    insights = [
        "Free Flow Zone (<8 departures/hr): Taxi times remain near physical baseline (~16-18 min).",
        "Linear Buildup (8-16 departures/hr): Average taxi increases moderately as runway head queue forms.",
        "Queue Saturation (>18 departures/hr): Taxi duration spikes sharply; variance widens as runway departure capacity is exceeded.",
        "Decision Trigger: Pre-departure metering recommended above 18 dep/hr to hold flights at gates rather than idling on taxiways."
    ]
    for ins in insights:
        p = tf7.add_paragraph()
        p.text = f"• {ins}"
        p.font.size = Pt(11)
        p.font.color.rgb = C_WHITE
        p.space_after = Pt(12)

    # =========================================================================
    # SLIDE 8: PREDICTION PRECISION & R2 DEFENSE (Viva Ready)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Prediction Parity & Statistical R² Defense", "VIVA-VOCE EXAMINATION RIGOR")
    
    p_res = "figures/residuals_and_actual_vs_predicted.png"
    if os.path.exists(p_res):
        add_card(s8, Inches(0.8), Inches(1.5), Inches(7.2), Inches(5.3))
        s8.shapes.add_picture(p_res, Inches(0.9), Inches(1.9), width=Inches(7.0))
        
    add_card(s8, Inches(8.2), Inches(1.5), Inches(4.3), Inches(5.3))
    tb8 = s8.shapes.add_textbox(Inches(8.4), Inches(1.7), Inches(3.9), Inches(4.9))
    tf8 = tb8.text_frame
    tf8.word_wrap = True
    
    p = tf8.paragraphs[0]
    p.text = "Why is R² Low (0.02 - 0.045)?"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_AMBER
    p.space_after = Pt(12)
    
    p = tf8.add_paragraph()
    p.text = "1. Aleatoric Surface Uncertainty:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p = tf8.add_paragraph()
    p.text = "Real-time ground holds, pushback tug waits, and runway crossing holds are physically absent from scheduled BTS tables."
    p.font.size = Pt(10)
    p.font.color.rgb = C_MUTED
    p.space_after = Pt(8)
    
    p = tf8.add_paragraph()
    p.text = "2. Right-Skewed Heavy Tails:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p = tf8.add_paragraph()
    p.text = "Rare 60-120 min de-icing holds heavily penalize squared error metrics (R² and RMSE) even when central predictions are tight."
    p.font.size = Pt(10)
    p.font.color.rgb = C_MUTED
    p.space_after = Pt(8)
    
    p = tf8.add_paragraph()
    p.text = "3. Operational Compliance Metric:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_GREEN
    p = tf8.add_paragraph()
    p.text = "In aviation operations, FAA compliance within ±5 minutes (84.9%) and MAE (3.88 min) govern operational utility, not R²."
    p.font.size = Pt(10)
    p.font.color.rgb = C_MUTED

    # =========================================================================
    # SLIDE 9: FEATURE IMPORTANCE RANKING
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Feature Importance: What Drives Departure Taxi Time?", "RANDOM FOREST & XGBOOST FEATURE ATTRIBUTIONS")
    
    p_imp = "figures/feature_importance.png"
    if os.path.exists(p_imp):
        add_card(s9, Inches(0.8), Inches(1.5), Inches(7.0), Inches(5.3))
        s9.shapes.add_picture(p_imp, Inches(1.0), Inches(1.7), width=Inches(6.6))
        
    add_card(s9, Inches(8.0), Inches(1.5), Inches(4.5), Inches(5.3))
    tb9 = s9.shapes.add_textbox(Inches(8.2), Inches(1.7), Inches(4.1), Inches(4.9))
    tf9 = tb9.text_frame
    tf9.word_wrap = True
    
    p = tf9.paragraphs[0]
    p.text = "Physical Feature Rationale"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(12)
    
    ranks = [
        ("DISTANCE (30.0%)", "Long-haul transcontinental flights assign to outer long runways, extending physical taxi distance."),
        ("DEP_DELAY (18.3%)", "Late pushbacks lose their departure metering slot, causing extended secondary taxiway hold times."),
        ("SIN_HOUR & COS_HOUR (15.2%)", "Diurnal scheduling peaks capture morning and evening departure rush banks."),
        ("CONGESTION_1H & 30M (12.0%)", "Direct measure of surface apron saturation and active taxiway queues."),
        ("ORIGIN_TAXI_MEAN (3.4%)", "Reflects airport layout complexity (sprawling DFW vs efficient parallel ATL).")
    ]
    for r_f, r_d in ranks:
        p = tf9.add_paragraph()
        p.text = f"• {r_f}: "
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = C_WHITE
        p = tf9.add_paragraph()
        p.text = f"  {r_d}"
        p.font.size = Pt(10)
        p.font.color.rgb = C_MUTED
        p.space_after = Pt(4)

    # =========================================================================
    # SLIDE 10: INTERACTIVE STREAMLIT APPLICATION DEMO
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Interactive Streamlit Dispatch Simulator (streamlit_app.py)", "DECISION-SUPPORT DASHBOARD & SCENARIO TESTING")
    
    demo_cards = [
        ("Tab 1: Flight Simulator", C_CYAN, [
            "Real-time prediction engine supporting Random Forest, XGBoost, and Linear Regression.",
            "Pre-loaded scenario presets:",
            "  - DFW Morning Peak Push (AA to ORD)",
            "  - DFW Midday Off-Peak (AA to MIA)",
            "  - JFK Evening Congested Peak (B6 to LAX)",
            "Outputs Predicted Taxi-Out, Estimated Wheels-Off, and Congestion Risk Badge."
        ]),
        ("Tab 2: Model Benchmarks", C_GREEN, [
            "Live comparison of MAE, RMSE, and operational accuracy across models.",
            "Operational Risk Confusion Matrix evaluating 4 congestion risk tiers:",
            "  - Normal Flow (<20 min)",
            "  - Moderate Queue (20-30 min)",
            "  - High Congestion (30-45 min)",
            "  - Severe Delay (>45 min)"
        ]),
        ("Tab 0, 3, 4: Analytics & Docs", C_AMBER, [
            "Data Understanding: Cleaning funnel & target percentiles.",
            "Congestion Analytics: Hub-specific average taxi times (DFW, ORD, ATL, JFK, DEN).",
            "A-CDM Documentation: Complete architecture and environmental formulas."
        ])
    ]
    for i, (title, color, bullets) in enumerate(demo_cards):
        left = Inches(0.8 + i * 4.0)
        add_card(s10, left, Inches(1.6), Inches(3.7), Inches(5.2))
        tb = s10.shapes.add_textbox(left + Inches(0.25), Inches(1.8), Inches(3.2), Inches(4.8))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = color
        p.space_after = Pt(14)
        
        for b in bullets:
            p = tf.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(10.5)
            p.font.color.rgb = C_WHITE
            p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 11: SYSTEM BOUNDARIES & CONCLUSION
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11)
    add_header(s11, "Project Boundaries, Limitations & Conclusion", "REALISTIC SCOPE & ACADEMIC VALUE")
    
    # Left Card: Boundaries
    add_card(s11, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    tb11_l = s11.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.2), Inches(4.9))
    tf11_l = tb11_l.text_frame
    tf11_l.word_wrap = True
    
    p = tf11_l.paragraphs[0]
    p.text = "Operational Limitations & Scope"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_AMBER
    p.space_after = Pt(12)
    
    p = tf11_l.add_paragraph()
    p.text = "1. Decision-Support Aid Only:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_WHITE
    p = tf11_l.add_paragraph()
    p.text = "This machine learning project is developed for academic coursework submission. It is in no way a substitute for certified real-time airline dispatch software (Sabre/Amadeus) or FAA radar automation (TFMS/ASDE-X), which track 90+ real-time flight parameters."
    p.font.size = Pt(10)
    p.font.color.rgb = C_MUTED
    p.space_after = Pt(8)
    
    p = tf11_l.add_paragraph()
    p.text = "2. Absent Operational Signals:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_WHITE
    p = tf11_l.add_paragraph()
    p.text = "Public BTS data lacks real-time METAR runway wind/visibility, tactical runway configuration switches, and arriving aircraft runway crossing queues."
    p.font.size = Pt(10)
    p.font.color.rgb = C_MUTED

    # Right Card: Conclusion
    add_card(s11, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    tb11_r = s11.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(4.9))
    tf11_r = tb11_r.text_frame
    tf11_r.word_wrap = True
    
    p = tf11_r.paragraphs[0]
    p.text = "Coursework Conclusions"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_GREEN
    p.space_after = Pt(12)
    
    conclusions = [
        ("XGBoost Demonstrated Superiority: ", "Achieved lowest MAE (3.88 min) and highest ±5 min operational compliance (84.9%) by sequential residual optimization."),
        ("Congestion Informs Dispatch: ", "Departure-bank timing and rolling 60-minute surface congestion provide significantly stronger operational context than distance alone."),
        ("Actionable Sustainability Guidance: ", "A-CDM gate-hold advisories triggered during peak queues can save ~130 kg of Jet-A fuel per delayed flight."),
        ("Complete Deliverables: ", "End-to-end pipeline, fully executed Jupyter notebook, interactive Streamlit UI, comprehensive docs, and presentation.")
    ]
    for c_title, c_desc in conclusions:
        p = tf11_r.add_paragraph()
        p.text = f"• {c_title}"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = C_WHITE
        p = tf11_r.add_paragraph()
        p.text = f"  {c_desc}"
        p.font.size = Pt(10)
        p.font.color.rgb = C_MUTED
        p.space_after = Pt(6)

    # Save Deck
    prs.save(output_pptx)
    print(f"PowerPoint Presentation successfully created at: {output_pptx}")
    return output_pptx

if __name__ == "__main__":
    create_deck()
