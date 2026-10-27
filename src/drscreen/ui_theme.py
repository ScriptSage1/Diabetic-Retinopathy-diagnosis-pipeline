"""Design system for the screening workstation: tokens, CSS, the small viewer script and the Gradio theme.

Everything visual lives here so the product name, palette, type scale, spacing and radii can be changed in one place. The HTML fragments are produced by
drscreen.explain; app/app.py only arranges them. Light theme only (a clinical reading environment): the script removes Gradio's dark class.

Rules the CSS follows: one hairline border colour, three radii, no shadows except for the expanded viewer layer, status colour is never the only signal
(every status also has a word, a shape or a position), motion is limited to short fades and switches off for `prefers-reduced-motion`.
"""
from __future__ import annotations

import os

PRODUCT_NAME = os.environ.get("DR_PRODUCT_NAME", "RetinaLab")
PRODUCT_DESCRIPTOR = "Diabetic Retinopathy Screening Research"
PROTOTYPE_TAG = "Research prototype · not for clinical use"

FONT_LINK = '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;450;550;600&display=swap">'

TOKENS = """
:root {
  --rl-font: 'Inter', 'Geist', 'Segoe UI Variable Text', 'Segoe UI', system-ui, -apple-system, Roboto, Helvetica, Arial, sans-serif;
  --rl-bg: #f5f3ee; --rl-surface: #fdfcf9; --rl-surface-2: #efede6; --rl-border: #dcd8cd; --rl-border-strong: #bdb8aa;
  --rl-text: #1c2220; --rl-text-2: #4a524e; --rl-text-3: #69716c;
  --rl-accent: #1f5f55; --rl-accent-hover: #174a42; --rl-accent-soft: #e3edea;
  --rl-refer: #a4412c; --rl-refer-soft: #f5e6e1; --rl-review: #80590f; --rl-review-soft: #f4ead3;
  --rl-ok: #2d6847; --rl-ok-soft: #e2eee6; --rl-neutral: #4e5963; --rl-neutral-soft: #e9ecee;
  --rl-viewer: #121615; --rl-viewer-text: #cfd5d2;
  --rl-r-sm: .1875rem; --rl-r: .375rem; --rl-r-lg: .5rem;
  --rl-s1: .25rem; --rl-s2: .5rem; --rl-s3: .75rem; --rl-s4: 1rem; --rl-s5: 1.5rem; --rl-s6: 2rem; --rl-s7: 3rem;
  --rl-gutter: clamp(1rem, 3.2vw, 4rem);
  --rl-fs-xs: .75rem; --rl-fs-sm: .8125rem; --rl-fs: .9375rem; --rl-fs-md: 1.0625rem; --rl-fs-lg: 1.3125rem; --rl-fs-xl: 1.75rem;
  --rl-ease: cubic-bezier(.2,.7,.2,1);
}
"""

CSS = TOKENS + """
/* ---------- page ---------- */
html, body, .gradio-container {background: var(--rl-bg) !important; color: var(--rl-text); font-family: var(--rl-font) !important; font-size: var(--rl-fs); -webkit-font-smoothing: antialiased;}
.gradio-container {max-width: none !important; width: 100% !important; margin: 0 !important; padding: 0 !important;}
.gradio-container .main {max-width: none !important; padding: 0 var(--rl-gutter) var(--rl-s6) var(--rl-gutter) !important;}
.gradio-container footer {display: none !important;}
.gradio-container .block, .gradio-container .form {box-shadow: none !important;}
.gradio-container .form {background: transparent !important; border: none !important; gap: var(--rl-s2) !important;}
.rl-plain, .rl-plain.block {background: transparent !important; border: none !important; padding: 0 !important; box-shadow: none !important;}
.rl-plain .html-container {padding: 0 !important;}
:focus-visible {outline: 2px solid var(--rl-accent) !important; outline-offset: 2px;}
.rl-sr {position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0; overflow: hidden; clip: rect(0 0 0 0); border: 0;}

/* ---------- header ---------- */
.rl-header {display: flex; align-items: center; justify-content: space-between; gap: var(--rl-s4); padding: var(--rl-s4) 0; border-bottom: 1px solid var(--rl-border); margin-bottom: var(--rl-s5); flex-wrap: wrap;}
.rl-brand {display: flex; align-items: center; gap: var(--rl-s3);} .rl-brand svg {flex: none; width: 1.9rem; height: 1.9rem;}
.rl-name {font-size: var(--rl-fs-md); font-weight: 600; letter-spacing: -.01em; line-height: 1.2; color: var(--rl-text);}
.rl-desc {font-size: var(--rl-fs-sm); color: var(--rl-text-2); line-height: 1.3;}
.rl-proto {font-size: var(--rl-fs-xs); color: var(--rl-text-2); border: 1px solid var(--rl-border-strong); border-radius: var(--rl-r-sm); padding: 3px 8px; letter-spacing: .01em;}
.rl-disclaimer {margin: var(--rl-s5) 0 0 0; padding-top: var(--rl-s3); border-top: 1px solid var(--rl-border); font-size: var(--rl-fs-xs); color: var(--rl-text-3); line-height: 1.5; max-width: 90ch;}

/* ---------- workspace ---------- */
#rl-workspace {gap: var(--rl-s5) !important; align-items: flex-start;}
.rl-eyebrow {font-size: var(--rl-fs-xs); font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--rl-text-3); margin: 0 0 var(--rl-s2) 0;}
.rl-h {font-size: var(--rl-fs-md); font-weight: 600; letter-spacing: -.005em; margin: 0 0 var(--rl-s3) 0; color: var(--rl-text);}
.rl-enter {animation: rl-fade .45s var(--rl-ease) both;}
@keyframes rl-fade {from {opacity: 0; transform: translateY(4px);} to {opacity: 1; transform: none;}}
@keyframes rl-slide {from {transform: translateX(-30%);} to {transform: translateX(130%);}}

/* ---------- input (before analysis) ---------- */
#rl-input {border: 1.5px dashed var(--rl-border-strong) !important; border-radius: var(--rl-r-lg) !important; background: var(--rl-surface) !important; transition: border-color .2s ease, background .2s ease;}
#rl-input:hover, #rl-input:focus-within {border-color: var(--rl-accent) !important; background: var(--rl-accent-soft) !important;}
#rl-input .image-container, #rl-input [data-testid="image"] {background: transparent !important;}
#rl-input label, #rl-input .label-wrap {color: var(--rl-text-2) !important;}
.rl-controls {gap: var(--rl-s2) !important; align-items: end !important; margin-top: var(--rl-s3);}
.rl-controls .block {padding: 0 !important; border: none !important; background: transparent !important;}
.gradio-container .rl-small-check {border: none !important; background: transparent !important; padding: 0 !important; margin-top: var(--rl-s2);}
.gradio-container .rl-accordion {border: none !important; border-top: 1px solid var(--rl-border) !important; border-radius: 0 !important; background: transparent !important; margin-top: var(--rl-s3);}
.gradio-container .rl-accordion > button, .gradio-container .rl-accordion .label-wrap {padding-left: 0 !important; font-size: var(--rl-fs-sm); color: var(--rl-text-2) !important;}
.gradio-container .rl-download {width: auto !important; min-width: 0 !important; align-self: flex-start; margin-top: var(--rl-s3);}
.rl-controls label span, .rl-controls .block-info {font-size: var(--rl-fs-xs) !important; font-weight: 600 !important; letter-spacing: .06em; text-transform: uppercase; color: var(--rl-text-3) !important;}
.gradio-container input[type=text], .gradio-container .wrap-inner, .gradio-container .secondary-wrap {border-radius: var(--rl-r) !important; background: var(--rl-surface) !important; border-color: var(--rl-border-strong) !important;}
.gradio-container button {border-radius: var(--rl-r) !important; font-weight: 550 !important; font-family: var(--rl-font) !important; transition: background .15s ease, border-color .15s ease, color .15s ease;}
.gradio-container button.primary {background: var(--rl-accent) !important; border: 1px solid var(--rl-accent) !important; color: #fff !important; box-shadow: none !important;}
.gradio-container button.primary:hover {background: var(--rl-accent-hover) !important; border-color: var(--rl-accent-hover) !important;}
.gradio-container button.secondary {background: var(--rl-surface) !important; border: 1px solid var(--rl-border-strong) !important; color: var(--rl-text) !important;}
.gradio-container button.secondary:hover {background: var(--rl-surface-2) !important;}
.gradio-container button[disabled] {opacity: .5; cursor: not-allowed;}
.gradio-container input[type=checkbox] {accent-color: var(--rl-accent);}
.gradio-container .rl-small-check label {font-size: var(--rl-fs-sm) !important; color: var(--rl-text-2) !important;}

/* ---------- evidence viewer ---------- */
.rl-viewer {background: var(--rl-surface); border: 1px solid var(--rl-border); border-radius: var(--rl-r-lg); overflow: hidden;}
.rl-viewer-bar {display: flex; align-items: center; justify-content: space-between; gap: var(--rl-s3); padding: var(--rl-s2) var(--rl-s3); border-bottom: 1px solid var(--rl-border); flex-wrap: wrap;}
.rl-seg {display: flex; gap: 2px; background: var(--rl-surface-2); border-radius: var(--rl-r); padding: 2px;}
.rl-seg label {padding: 5px 12px; font-size: var(--rl-fs-sm); font-weight: 500; color: var(--rl-text-2); border-radius: var(--rl-r-sm); cursor: pointer; white-space: nowrap; transition: background .15s ease, color .15s ease;}
.rl-seg label:hover {color: var(--rl-text);}
.rl-seg label.rl-off {opacity: .4; cursor: not-allowed; text-decoration: none;}
#rlv-orig:checked ~ .rl-viewer-bar label[for=rlv-orig], #rlv-proc:checked ~ .rl-viewer-bar label[for=rlv-proc],
#rlv-cam:checked ~ .rl-viewer-bar label[for=rlv-cam], #rlv-les:checked ~ .rl-viewer-bar label[for=rlv-les] {background: var(--rl-surface); color: var(--rl-accent); font-weight: 600; box-shadow: inset 0 -2px 0 var(--rl-accent);}
#rlv-orig:focus-visible ~ .rl-viewer-bar label[for=rlv-orig], #rlv-proc:focus-visible ~ .rl-viewer-bar label[for=rlv-proc],
#rlv-cam:focus-visible ~ .rl-viewer-bar label[for=rlv-cam], #rlv-les:focus-visible ~ .rl-viewer-bar label[for=rlv-les] {outline: 2px solid var(--rl-accent); outline-offset: 1px;}
.rl-tools {display: flex; gap: 2px;}
.gradio-container .rl-tools button {width: 1.875rem; height: 1.875rem; padding: 0 !important; min-width: 0 !important; display: inline-flex; align-items: center; justify-content: center; background: transparent !important; border: 1px solid transparent !important; color: var(--rl-text-2) !important;}
.gradio-container .rl-tools button:hover {background: var(--rl-surface-2) !important; color: var(--rl-text) !important;}
.rl-tools svg {width: 1rem; height: 1rem; stroke: currentColor; fill: none; stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round;}
.rl-stage-wrap {background: var(--rl-viewer);}
.rl-view {display: none;}
#rlv-orig:checked ~ .rl-stage-wrap [data-view=orig], #rlv-proc:checked ~ .rl-stage-wrap [data-view=proc], #rlv-cam:checked ~ .rl-stage-wrap [data-view=cam], #rlv-les:checked ~ .rl-stage-wrap [data-view=les] {display: block;}
.rl-stage {width: 100%; aspect-ratio: 4 / 3; max-height: 72vh; overflow: hidden; display: flex; align-items: center; justify-content: center; touch-action: none; user-select: none;}
.rl-img {width: 100%; height: 100%; object-fit: contain; transform-origin: center; will-change: transform; cursor: zoom-in; -webkit-user-drag: none;}
.rl-img.rl-zoomed {cursor: grab;} .rl-img.rl-panning {cursor: grabbing;}
.rl-viewer-foot {display: flex; justify-content: space-between; gap: var(--rl-s4); padding: var(--rl-s2) var(--rl-s3); border-top: 1px solid var(--rl-border); font-size: var(--rl-fs-xs); color: var(--rl-text-2); line-height: 1.45; flex-wrap: wrap;}
.rl-cap {display: none; max-width: 70ch;}
#rlv-orig:checked ~ .rl-viewer-foot [data-cap=orig], #rlv-proc:checked ~ .rl-viewer-foot [data-cap=proc], #rlv-cam:checked ~ .rl-viewer-foot [data-cap=cam], #rlv-les:checked ~ .rl-viewer-foot [data-cap=les] {display: block;}
.rl-viewer.rl-expanded {position: fixed; inset: var(--rl-s5); z-index: 1000; display: flex; flex-direction: column; box-shadow: 0 0 0 100vmax rgba(18,22,21,.62);}
.rl-viewer.rl-expanded .rl-stage-wrap {flex: 1; min-height: 0; position: relative;}
.rl-viewer.rl-expanded .rl-view {position: absolute; inset: 0;} .rl-viewer.rl-expanded .rl-stage {height: 100%; aspect-ratio: auto; max-height: none;}

/* ---------- result panel ---------- */
.rl-panel {border: 1px solid var(--rl-border); border-radius: var(--rl-r-lg); background: var(--rl-surface); padding: var(--rl-s5);}
.rl-reco {font-size: var(--rl-fs-xl); font-weight: 600; letter-spacing: -.02em; line-height: 1.15; margin: 0; color: var(--accent, var(--rl-text));}
.rl-reco-sub {margin: var(--rl-s2) 0 0 0; color: var(--rl-text-2); line-height: 1.5; max-width: 52ch;}
.rl-kv {margin: var(--rl-s4) 0 0 0; border-top: 1px solid var(--rl-border);}
.rl-kv > div {display: flex; justify-content: space-between; gap: var(--rl-s4); padding: 9px 0; border-bottom: 1px solid var(--rl-border); font-size: var(--rl-fs); align-items: baseline;}
.rl-kv dt {color: var(--rl-text-2); margin: 0; white-space: nowrap; flex: none;} .rl-kv dd {margin: 0; min-width: 0; text-align: right; font-weight: 500; font-variant-numeric: tabular-nums; color: var(--rl-text);}
.rl-kv dd small {display: block; font-weight: 400; font-size: var(--rl-fs-xs); color: var(--rl-text-3);}
.rl-note {margin: var(--rl-s4) 0 0 0; font-size: var(--rl-fs-sm); line-height: 1.55; color: var(--rl-text-2); max-width: 62ch;}
.rl-score {margin-top: var(--rl-s5);} .rl-score svg {display: block; width: 100%; max-width: 36rem; height: auto; overflow: visible;}
.rl-score text {font-family: var(--rl-font); font-size: .9rem; fill: var(--rl-text-2);} .rl-score .rl-lab-strong {fill: var(--rl-text); font-weight: 600;}
.rl-score .rl-axis {stroke: var(--rl-border-strong); stroke-width: 2;} .rl-score .rl-tick {stroke: var(--rl-border-strong); stroke-width: 1;}
.rl-score .rl-thr {stroke: var(--rl-text); stroke-width: 1.5; stroke-dasharray: 3 3;} .rl-score .rl-marker {fill: var(--accent, var(--rl-text)); animation: rl-fade .6s var(--rl-ease) .2s both;}
.rl-score .rl-zone-ref {fill: var(--rl-refer); opacity: .09;} .rl-score .rl-band {fill: url(#rl-hatch);}
.rl-loading .rl-bar {height: 2px; background: var(--rl-border); border-radius: 2px; overflow: hidden; margin-top: var(--rl-s4); position: relative;}
.rl-loading .rl-bar::after {content: ''; position: absolute; inset: 0 auto 0 0; width: 35%; background: var(--rl-accent); animation: rl-slide 1.3s ease-in-out infinite;}
.rl-empty ol {margin: var(--rl-s3) 0 0 0; padding-left: 1.2em; color: var(--rl-text-2); line-height: 1.7;} .rl-empty .rl-reco {color: var(--rl-text);}
.rl-empty p {margin: var(--rl-s2) 0 0 0; color: var(--rl-text-2); line-height: 1.55; max-width: 52ch;}

/* ---------- detail section ---------- */
#rl-details {margin-top: var(--rl-s6);}
#rl-details .tab-wrapper, #rl-details .tab-container {border-bottom: 1px solid var(--rl-border) !important;}
#rl-details button[role=tab] {background: transparent !important; border: none !important; border-bottom: 2px solid transparent !important; border-radius: 0 !important; color: var(--rl-text-2) !important; padding: 10px 14px !important; font-weight: 500 !important;}
#rl-details button[role=tab][aria-selected=true] {color: var(--rl-accent) !important; border-bottom-color: var(--rl-accent) !important; font-weight: 600 !important;}
#rl-details button[role=tab] {text-overflow: clip !important; overflow: visible !important; max-width: none !important; white-space: nowrap;}
#rl-details .tabitem, #rl-details [role=tabpanel] {border: none !important; background: transparent !important; padding: var(--rl-s5) 0 0 0 !important;}
.rl-prose {max-width: 74ch; line-height: 1.65;} .rl-prose ul {margin: 0; padding-left: 1.1em;} .rl-prose li {margin: 0 0 var(--rl-s3) 0;}
.rl-prose li::marker {color: var(--rl-text-3);}
.rl-figs {display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 16rem), 28rem)); gap: var(--rl-s4); margin-bottom: var(--rl-s4);}
.rl-fig {margin: 0;} .rl-fig img {display: block; width: 100%; height: auto; border-radius: var(--rl-r); background: var(--rl-viewer); border: 1px solid var(--rl-border);}
.rl-fig figcaption {margin-top: var(--rl-s2); font-size: var(--rl-fs-xs); color: var(--rl-text-2); line-height: 1.5;}
.rl-table, .rl-table tr, .rl-table th, .rl-table td {border-left: none !important; border-right: none !important; border-top: none !important; background: transparent !important;}
.rl-table {width: 100%; border-collapse: collapse; font-size: var(--rl-fs-sm); border: none !important;}
.rl-table td:first-child {width: 34%;} .rl-table td:nth-child(2) {width: 1%; white-space: nowrap;} .rl-table th {text-align: left; font-weight: 600; font-size: var(--rl-fs-xs); letter-spacing: .05em; text-transform: uppercase; color: var(--rl-text-3); padding: 8px 12px 8px 0; border-bottom: 1px solid var(--rl-border-strong);}
.rl-table td {padding: 9px 12px 9px 0; border-bottom: 1px solid var(--rl-border); vertical-align: top; line-height: 1.5; color: var(--rl-text);}
.rl-table .num {text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap;}
.rl-status {display: inline-block; font-size: var(--rl-fs-xs); font-weight: 600; padding: 1px 7px; border: 1px solid currentColor; border-radius: var(--rl-r-sm); white-space: nowrap;}
.rl-status.met {color: var(--rl-refer);} .rl-status.not_met {color: var(--rl-ok);} .rl-status.not_assessed {color: var(--rl-neutral);}
.rl-counts {display: grid; grid-template-columns: repeat(auto-fit, minmax(7rem, 1fr)); gap: 0; border: 1px solid var(--rl-border); border-radius: var(--rl-r); margin-bottom: var(--rl-s4);}
.rl-counts > div {padding: var(--rl-s3) var(--rl-s4); border-right: 1px solid var(--rl-border);} .rl-counts > div:last-child {border-right: none;}
.rl-counts b {display: block; font-size: var(--rl-fs-lg); font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1.2;} .rl-counts span {font-size: var(--rl-fs-xs); color: var(--rl-text-2);}
.rl-quad {border-collapse: collapse; margin: var(--rl-s2) 0 0 0;} .rl-quad td {border: 1px solid var(--rl-border-strong); padding: 8px 14px; font-size: var(--rl-fs-xs); text-align: center; color: var(--rl-text-2); line-height: 1.5;} .rl-quad b {color: var(--rl-text); font-weight: 600;}
.rl-prob {max-width: 48rem; display: grid; grid-template-columns: minmax(9rem, 12rem) 1fr 3.5rem; gap: var(--rl-s3); align-items: center; padding: 7px 0; border-bottom: 1px solid var(--rl-border); font-size: var(--rl-fs-sm);}
.rl-prob .track {height: 6px; background: var(--rl-surface-2); border-radius: 3px; overflow: hidden;} .rl-prob .track i {display: block; height: 100%; width: var(--w); background: var(--rl-text-3); border-radius: 3px; animation: rl-grow .8s var(--rl-ease) .1s both;}
.rl-prob.top .track i {background: var(--accent, var(--rl-accent));} .rl-prob.top {font-weight: 600;} .rl-prob .v {text-align: right; font-variant-numeric: tabular-nums;}
@keyframes rl-grow {from {width: 0;} to {width: var(--w);}}
.rl-small {font-size: var(--rl-fs-xs); color: var(--rl-text-3); line-height: 1.55; margin-top: var(--rl-s3); max-width: 80ch;}
.rl-sub {font-size: var(--rl-fs-sm); font-weight: 600; margin: var(--rl-s5) 0 var(--rl-s2) 0; color: var(--rl-text);}

/* ---------- presentation layer: headline numbers, details, motion ---------- */
.rl-stats {display: grid; grid-template-columns: repeat(3, 1fr); margin-top: var(--rl-s5); border-top: 1px solid var(--rl-border); border-bottom: 1px solid var(--rl-border);}
.rl-stats > div {padding: var(--rl-s4) var(--rl-s3) var(--rl-s3) 0; animation: rl-rise .6s var(--rl-ease) both;}
.rl-stats > div + div {border-left: 1px solid var(--rl-border); padding-left: var(--rl-s4);}
.rl-stats > div:nth-child(2) {animation-delay: .08s;} .rl-stats > div:nth-child(3) {animation-delay: .16s;}
.rl-stat-v {display: block; font-size: 1.85rem; font-weight: 600; letter-spacing: -.02em; line-height: 1.1; color: var(--rl-text); font-variant-numeric: tabular-nums;}
.rl-stat-v.rl-count {color: var(--accent, var(--rl-text));} .rl-stat-v.rl-stat-word {font-size: 1.3rem; padding-top: .35rem;}
.rl-stat-l {display: block; margin-top: .3rem; font-size: var(--rl-fs-xs); letter-spacing: .04em; color: var(--rl-text-3);}
.rl-result .rl-reco {animation: rl-rise .55s var(--rl-ease) both;} .rl-result .rl-reco-sub {animation: rl-rise .55s var(--rl-ease) .06s both;}
.rl-result .rl-score {animation: rl-fade .6s ease .2s both;}
.rl-score .rl-mk {animation: rl-mk 1.15s cubic-bezier(.2,.85,.25,1) .35s both;}
.rl-score .rl-marker {animation: none;}
.rl-score .rl-zone-ref, .rl-score .rl-band {transform-box: fill-box; transform-origin: left center; animation: rl-grow-x .9s var(--rl-ease) .2s both;}
@keyframes rl-mk {from {transform: translateX(var(--dx)); opacity: 0;} 25% {opacity: 1;} to {transform: none; opacity: 1;}}
@keyframes rl-grow-x {from {transform: scaleX(0);} to {transform: scaleX(1);}}
@keyframes rl-rise {from {opacity: 0; transform: translateY(8px);} to {opacity: 1; transform: none;}}
.rl-more {margin-top: var(--rl-s4); border-top: 1px solid var(--rl-border); animation: rl-fade .6s ease .45s both;}
.rl-more summary {cursor: pointer; list-style: none; padding: var(--rl-s3) 0; font-size: var(--rl-fs-sm); font-weight: 550; color: var(--rl-text-2); display: flex; align-items: center; gap: var(--rl-s2);}
.rl-more summary::-webkit-details-marker {display: none;}
.rl-more summary::before {content: ''; width: .45rem; height: .45rem; border-right: 1.6px solid currentColor; border-bottom: 1.6px solid currentColor; transform: rotate(-45deg); transition: transform .25s var(--rl-ease);}
.rl-more[open] summary::before {transform: rotate(45deg);}
.rl-more summary:hover {color: var(--rl-text);}
.rl-more[open] .rl-kv {animation: rl-rise .35s var(--rl-ease) both;} .rl-more .rl-kv {margin-top: 0;}
.rl-kv dt[title] {text-decoration: underline dotted var(--rl-border-strong); text-underline-offset: 3px; cursor: help;}
.rl-view {animation: rl-fade .35s ease both;}
.rl-viewer .rl-img {animation: rl-settle .6s var(--rl-ease) both;}
@keyframes rl-settle {from {opacity: 0; transform: scale(1.015);} to {opacity: 1; transform: none;}}
.rl-scanning .rl-viewer-bar {padding: var(--rl-s3);}
.rl-scan-label {display: inline-flex; align-items: center; gap: var(--rl-s2); font-size: var(--rl-fs-sm); color: var(--rl-text-2);}
.rl-dot {width: .55rem; height: .55rem; border-radius: 50%; background: var(--rl-accent); animation: rl-pulse 1.2s ease-in-out infinite;}
@keyframes rl-pulse {0%, 100% {opacity: .35; transform: scale(.85);} 50% {opacity: 1; transform: scale(1);}}
.rl-stage-scan {position: relative;}
.rl-img-scan {filter: saturate(.85) brightness(.92); cursor: default; animation: none;}
.rl-scanline {position: absolute; left: 0; right: 0; top: 0; height: 22%; pointer-events: none;
  background: linear-gradient(to bottom, rgba(120, 220, 200, 0) 0%, rgba(120, 220, 200, .16) 80%, rgba(170, 255, 235, .65) 100%);
  border-bottom: 1px solid rgba(190, 255, 240, .85); animation: rl-scan 2.2s cubic-bezier(.45, 0, .55, 1) infinite;}
@keyframes rl-scan {from {transform: translateY(-100%);} to {transform: translateY(460%);}}
.gradio-container button.primary {transition: background .2s ease, transform .2s var(--rl-ease), box-shadow .2s ease !important;}
.gradio-container button.primary:hover {transform: translateY(-1px); box-shadow: 0 6px 16px rgba(31, 95, 85, .22) !important;}
.gradio-container button.primary:active {transform: none;}
.rl-header {animation: rl-fade .6s ease both;}

/* ---------- responsive ---------- */
@media (max-width: 54em) {
  #rl-workspace {flex-direction: column !important;} #rl-workspace > div {width: 100% !important; min-width: 0 !important;}
  .rl-stat-v {font-size: 1.45rem;} .rl-stat-v.rl-stat-word {font-size: 1.05rem;}
  .rl-kv > div {flex-direction: column; gap: .125rem;} .rl-kv dd {text-align: left;} .rl-score text {font-size: 1.2rem;}
  .rl-prob {grid-template-columns: 7.5rem 1fr 3.25rem;} .rl-panel {padding: var(--rl-s4);} .rl-reco {font-size: 1.45rem;}
  .rl-viewer.rl-expanded {inset: var(--rl-s2);}
}
@media (prefers-reduced-motion: reduce) {*, *::before, *::after {animation: none !important; transition: none !important;} .rl-prob .track i {width: var(--w) !important;}}
"""

# Viewer behaviour. The viewer HTML is replaced on every analysis, so listeners are delegated from the document.
JS = r"""
(function () {
  var lock = function () { document.documentElement.classList.remove('dark'); if (document.body) document.body.classList.remove('dark'); };
  lock();
  // the script can be injected after DOMContentLoaded has already fired, so run at once in that case
  var onReady = function (fn) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn); else fn(); };
  onReady(function () {
    lock();
    new MutationObserver(lock).observe(document.body, {attributes: true, attributeFilter: ['class']});
  });
  var MIN = 1, MAX = 8;
  function activeImg(v) {
    var views = v.querySelectorAll('.rl-view');
    for (var i = 0; i < views.length; i++) { if (views[i].offsetParent !== null) return views[i].querySelector('.rl-img'); }
    return null;
  }
  function paint(img) {
    var s = +img.dataset.s || 1, x = +img.dataset.x || 0, y = +img.dataset.y || 0;
    var bx = (s - 1) * img.clientWidth / 2, by = (s - 1) * img.clientHeight / 2;
    x = Math.max(-bx, Math.min(bx, x)); y = Math.max(-by, Math.min(by, y));
    img.dataset.x = x; img.dataset.y = y;
    img.style.transform = 'translate(' + x + 'px,' + y + 'px) scale(' + s + ')';
    img.classList.toggle('rl-zoomed', s > 1.001);
  }
  function zoom(img, factor, reset) {
    var s = reset ? 1 : Math.max(MIN, Math.min(MAX, (+img.dataset.s || 1) * factor));
    img.dataset.s = s; if (s === 1) { img.dataset.x = 0; img.dataset.y = 0; }
    paint(img);
  }
  document.addEventListener('click', function (e) {
    var b = e.target.closest('[data-rl]'); if (!b) return;
    var v = b.closest('.rl-viewer'); if (!v) return;
    var act = b.getAttribute('data-rl'), img = activeImg(v);
    if (act === 'expand') { v.classList.toggle('rl-expanded'); b.setAttribute('aria-pressed', v.classList.contains('rl-expanded')); if (img) zoom(img, 1, true); return; }
    if (!img) return;
    if (act === 'in') zoom(img, 1.4); else if (act === 'out') zoom(img, 1 / 1.4); else zoom(img, 1, true);
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') { var v = document.querySelector('.rl-viewer.rl-expanded'); if (v) v.classList.remove('rl-expanded'); }
  });
  document.addEventListener('dblclick', function (e) {
    var img = e.target.closest('.rl-img'); if (!img) return;
    zoom(img, 1, (+img.dataset.s || 1) > 1.001); if ((+img.dataset.s || 1) <= 1.001 && !img.dataset.dz) { zoom(img, 2.5); }
  });
  document.addEventListener('wheel', function (e) {
    var img = e.target.closest('.rl-img'); if (!img || !(e.ctrlKey || e.metaKey)) return;
    e.preventDefault(); zoom(img, e.deltaY < 0 ? 1.15 : 1 / 1.15);
  }, {passive: false});
  // headline numbers count up from zero when a new result appears
  var countUp = function (el) {
    el.dataset.done = '1';
    var to = parseFloat(el.dataset.to), dec = (el.dataset.to.split('.')[1] || '').length, t0 = null, dur = 900;
    if (isNaN(to) || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    var step = function (ts) {
      if (t0 === null) t0 = ts;
      var k = Math.min(1, (ts - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = (to * e).toFixed(dec);
      if (k < 1) requestAnimationFrame(step); else el.textContent = to.toFixed(dec);
    };
    el.textContent = (0).toFixed(dec);
    requestAnimationFrame(step);
  };
  var scanCounts = function () { document.querySelectorAll('.rl-count:not([data-done])').forEach(countUp); };
  onReady(function () {
    scanCounts();
    new MutationObserver(scanCounts).observe(document.body, {childList: true, subtree: true});
  });
  var drag = null;
  document.addEventListener('pointerdown', function (e) {
    var img = e.target.closest('.rl-img'); if (!img || (+img.dataset.s || 1) <= 1.001) return;
    drag = {img: img, px: e.clientX, py: e.clientY, x: +img.dataset.x || 0, y: +img.dataset.y || 0}; img.classList.add('rl-panning'); img.setPointerCapture(e.pointerId);
  });
  document.addEventListener('pointermove', function (e) {
    if (!drag) return; drag.img.dataset.x = drag.x + e.clientX - drag.px; drag.img.dataset.y = drag.y + e.clientY - drag.py; paint(drag.img);
  });
  var end = function () { if (drag) { drag.img.classList.remove('rl-panning'); drag = null; } };
  document.addEventListener('pointerup', end); document.addEventListener('pointercancel', end);
})();
"""

HEAD = FONT_LINK + "<script>" + JS + "</script>"

_MARK = ('<svg width="30" height="30" viewBox="0 0 32 32" fill="none" aria-hidden="true"><circle cx="16" cy="16" r="13" stroke="#1f5f55" stroke-width="2"/>'
         '<circle cx="16" cy="16" r="6.5" stroke="#1f5f55" stroke-width="2"/><circle cx="19.2" cy="13.2" r="2.1" fill="#1f5f55"/></svg>')


def header_html() -> str:
    from . import INTENDED_USE  # noqa: F401  (kept importable for callers that want the full statement)

    return (f'<header class="rl-header"><div class="rl-brand">{_MARK}<div><div class="rl-name">{PRODUCT_NAME}</div>'
            f'<div class="rl-desc">{PRODUCT_DESCRIPTOR}</div></div></div><div class="rl-proto">{PROTOTYPE_TAG}</div></header>')


def disclaimer_html() -> str:
    from . import INTENDED_USE

    return f'<p class="rl-disclaimer">{INTENDED_USE}</p>'


def build_theme():
    """Gradio theme: maps the design tokens onto Gradio's variables so native widgets (dropdown, upload, tabs) match the custom HTML."""
    import gradio as gr

    hue = gr.themes.Color(c50="#eef5f3", c100="#e3edea", c200="#c5dbd5", c300="#9cc2b9", c400="#5f9a8d", c500="#2f7467", c600="#1f5f55", c700="#174a42",
                          c800="#123a34", c900="#0d2b27", c950="#081d1a", name="forest_teal")
    t = gr.themes.Base(primary_hue=hue, secondary_hue="stone", neutral_hue="stone", radius_size=gr.themes.sizes.radius_sm, text_size=gr.themes.sizes.text_md,
                       font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "system-ui", "sans-serif"])
    kw = dict(body_background_fill="#f5f3ee", body_background_fill_dark="#f5f3ee", body_text_color="#1c2220", body_text_color_dark="#1c2220",
              background_fill_primary="#fdfcf9", background_fill_primary_dark="#fdfcf9", background_fill_secondary="#f5f3ee", background_fill_secondary_dark="#f5f3ee",
              block_background_fill="#fdfcf9", block_background_fill_dark="#fdfcf9", block_border_width="1px", block_border_color="#dcd8cd", block_border_color_dark="#dcd8cd",
              block_radius="8px", block_shadow="none", block_shadow_dark="none", block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
              block_label_text_color="#4a524e", block_label_text_color_dark="#4a524e", block_title_text_color="#4a524e", block_title_text_color_dark="#4a524e",
              border_color_primary="#dcd8cd", border_color_primary_dark="#dcd8cd", input_background_fill="#fdfcf9", input_background_fill_dark="#fdfcf9",
              input_border_color="#bdb8aa", input_border_color_dark="#bdb8aa", input_radius="6px", input_shadow="none", input_shadow_dark="none",
              button_primary_background_fill="#1f5f55", button_primary_background_fill_dark="#1f5f55", button_primary_background_fill_hover="#174a42",
              button_primary_background_fill_hover_dark="#174a42", button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
              button_primary_border_color="#1f5f55", button_primary_border_color_dark="#1f5f55",
              button_secondary_background_fill="#fdfcf9", button_secondary_background_fill_dark="#fdfcf9", button_secondary_background_fill_hover="#efede6",
              button_secondary_background_fill_hover_dark="#efede6", button_secondary_text_color="#1c2220", button_secondary_text_color_dark="#1c2220",
              button_secondary_border_color="#bdb8aa", button_secondary_border_color_dark="#bdb8aa", button_large_radius="6px", button_medium_radius="6px", button_small_radius="6px",
              checkbox_background_color_selected="#1f5f55", checkbox_background_color_selected_dark="#1f5f55")
    return t.set(**kw)
