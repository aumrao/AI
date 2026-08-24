import streamlit as st


def load_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Global Reset & Typography */
        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            color: #0f172a;
            background-color: #f8fafc;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 1.5rem;
            padding-bottom: 4rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Hero Header */
        .app-hero-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 60%, #312e81 100%);
            border-radius: 20px;
            padding: 24px 32px;
            margin-bottom: 24px;
            box-shadow: 0 12px 30px -8px rgba(15, 23, 42, 0.25);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .hero-left {
            display: flex;
            align-items: center;
            gap: 18px;
        }

        .hero-logo-box {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 52px;
            height: 52px;
            border-radius: 14px;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            font-size: 26px;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        }

        .hero-title-group h1 {
            font-size: 24px !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px;
            margin: 0 !important;
            color: #ffffff !important;
            line-height: 1.2;
        }

        .hero-title-group p {
            font-size: 13px !important;
            color: #cbd5e1 !important;
            margin: 3px 0 0 0 !important;
            font-weight: 400;
        }

        .hero-status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border-radius: 30px;
            font-size: 12px;
            font-weight: 600;
            color: #e2e8f0;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #10b981;
            box-shadow: 0 0 8px #10b981;
            animation: pulse-dot 2s infinite;
        }

        @keyframes pulse-dot {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.85); }
        }

        /* Config Card Container */
        .config-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04);
        }

        .config-section-title {
            font-size: 14px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #64748b;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .section-grid {
            display: grid;
            grid-template-columns: 200px 1fr;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            overflow: hidden;
            background: #ffffff;
        }

        .section-label {
            display: flex;
            align-items: center;
            padding: 18px 20px;
            font-size: 14px;
            font-weight: 600;
            color: #334155;
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
            border-bottom: 1px solid #e2e8f0;
        }

        .section-label:last-child {
            border-bottom: none;
        }

        .section-input-cell {
            padding: 12px 18px;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            align-items: center;
        }

        .section-input-cell:last-child {
            border-bottom: none;
        }

        /* Modern Input TextAreas */
        .stTextArea textarea {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            font-size: 14px !important;
            border: 1.5px solid #e2e8f0 !important;
            border-radius: 10px !important;
            background: #fdfdfd !important;
            color: #0f172a !important;
            transition: all 0.2s ease !important;
            padding: 10px 14px !important;
        }

        .stTextArea textarea:focus {
            border-color: #6366f1 !important;
            background: #ffffff !important;
            box-shadow: 0 0 0 3.5px rgba(99, 102, 241, 0.15) !important;
        }

        /* Modern Radio Buttons */
        div[data-testid="stRadio"] > div {
            gap: 1.5rem;
        }

        div[data-testid="stRadio"] label {
            font-size: 14px !important;
            font-weight: 500 !important;
            color: #1e293b !important;
            cursor: pointer;
        }

        /* Action Buttons Row */
        .action-row {
            margin-bottom: 24px;
        }

        .action-row button[kind="primary"],
        .action-row button[kind="secondary"] {
            min-height: 90px !important;
            border-radius: 16px !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            padding: 18px 24px !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            white-space: normal !important;
            text-align: center !important;
        }

        .action-row button[kind="primary"] {
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: 0 8px 20px -4px rgba(79, 70, 229, 0.4) !important;
        }

        .action-row button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 26px -4px rgba(79, 70, 229, 0.5) !important;
            background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%) !important;
        }

        .action-row button[kind="secondary"] {
            background: #ffffff !important;
            color: #1e293b !important;
            border: 1.5px solid #e2e8f0 !important;
            box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.04) !important;
        }

        .action-row button[kind="secondary"]:hover {
            border-color: #6366f1 !important;
            color: #4f46e5 !important;
            background: #f5f7ff !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px -4px rgba(99, 102, 241, 0.15) !important;
        }

        /* Dynamic Card Wrapper */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #e2e8f0 !important;
            border-radius: 18px !important;
            padding: 24px !important;
            background: #ffffff !important;
            box-shadow: 0 10px 30px -5px rgba(15, 23, 42, 0.06) !important;
            margin-bottom: 24px !important;
            animation: fadeIn 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Badges */
        .card-header-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            font-weight: 700;
            padding: 6px 16px;
            border-radius: 30px;
            margin-bottom: 12px;
        }

        .badge-interview {
            background: linear-gradient(135deg, #4338ca 0%, #6366f1 100%);
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
        }

        .badge-summary {
            background: linear-gradient(135deg, #0f172a 0%, #334155 100%);
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.2);
        }

        .audio-controls-label {
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: #64748b;
            margin-top: 18px;
            margin-bottom: 8px;
        }

        .audio-section-box {
            border: 1.5px dashed #c7d2fe;
            border-radius: 14px;
            padding: 20px;
            background: linear-gradient(135deg, #f8faff 0%, #f1f5f9 100%);
            margin-top: 14px;
        }

        /* Transcript Card */
        .transcript-card {
            border: 1.5px solid #10b981;
            border-radius: 16px;
            padding: 20px 24px;
            background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
            box-shadow: 0 4px 18px -2px rgba(16, 185, 129, 0.15);
            margin-top: 18px;
            animation: fadeIn 0.3s ease-out;
        }

        .transcript-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            flex-wrap: wrap;
            gap: 8px;
        }

        .transcript-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            font-weight: 700;
            padding: 5px 12px;
            border-radius: 20px;
            background: #10b981;
            color: #ffffff;
        }

        .transcript-metric {
            font-size: 12px;
            font-weight: 600;
            color: #047857;
            background: rgba(16, 185, 129, 0.12);
            padding: 4px 10px;
            border-radius: 20px;
        }

        .transcript-text-display {
            font-size: 15px;
            line-height: 1.6;
            color: #0f172a;
            padding: 12px 16px;
            background: #ffffff;
            border-radius: 10px;
            border-left: 4px solid #10b981;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #10b981;
            margin-bottom: 14px;
            white-space: pre-wrap;
        }

        /* Evaluation Report Card */
        .eval-report-card {
            border: 1.5px solid #818cf8;
            border-radius: 14px;
            padding: 16px 20px;
            background: linear-gradient(135deg, #f5f3ff 0%, #ffffff 100%);
            margin-top: 14px;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.08);
            animation: fadeIn 0.3s ease-out;
        }

        .eval-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 20px;
            background: #6366f1;
            color: #ffffff;
            margin-bottom: 8px;
        }

        /* Candidate Performance Evaluation Card */
        .candidate-eval-card {
            border: 2px solid #818cf8;
            border-radius: 16px;
            padding: 20px 24px;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            margin-top: 18px;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.12), 0 8px 10px -6px rgba(99, 102, 241, 0.08);
            animation: fadeIn 0.4s ease-out;
        }

        .candidate-eval-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1.5px solid #e2e8f0;
        }

        .eligibility-badge-eligible {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 14px;
            font-weight: 700;
            padding: 6px 16px;
            border-radius: 20px;
            background: #ecfdf5;
            color: #065f46;
            border: 1.5px solid #34d399;
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.2);
        }

        .eligibility-badge-borderline {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 14px;
            font-weight: 700;
            padding: 6px 16px;
            border-radius: 20px;
            background: #fffbeb;
            color: #92400e;
            border: 1.5px solid #f59e0b;
        }

        .eligibility-badge-rejected {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 14px;
            font-weight: 700;
            padding: 6px 16px;
            border-radius: 20px;
            background: #fef2f2;
            color: #991b1b;
            border: 1.5px solid #ef4444;
        }

        .prob-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 14px;
            font-weight: 700;
            padding: 6px 16px;
            border-radius: 20px;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
        }

        .eval-report-content {
            font-size: 14px;
            line-height: 1.65;
            color: #1e293b;
            background: #ffffff;
            padding: 16px 20px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
        }




        /* Responsive Layout */
        @media (max-width: 800px) {
            .app-hero-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 14px;
                padding: 20px;
            }

            .section-grid {
                grid-template-columns: 1fr;
            }

            .section-label {
                border-right: none;
                border-bottom: 1px solid #e2e8f0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


