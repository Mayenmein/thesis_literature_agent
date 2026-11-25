from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.getenv("THESIS_LATEX_OUTPUT_DIR", str(BASE_DIR / "thesis_output"))).resolve()
DEFAULT_TARGET_FILENAME = "Mayenmein_Terence_Masters_generated.tex"

# Tracks whether the single allowed correction attempt has already been used per target + section_key
_CORRECTION_ATTEMPTS_USED: set[str] = set()

EXACT_THESIS_PREAMBLE = r"""\documentclass[openany,a4paper,12pt]{book}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\usepackage{amsmath}
\usepackage{ragged2e}
\usepackage{amsthm}
\justifying
\usepackage{graphicx}
\usepackage{float}
\usepackage{adjustbox}
\usepackage{pgfplots}
\usepackage{filecontents} 
\usepackage{setspace}
\usepackage[colorlinks=true, linkcolor=black, citecolor=black, urlcolor=black]{hyperref}
\usepackage{multirow}
\usepackage{booktabs}
\usepackage{caption} 
\usepackage{tikz}
\usetikzlibrary{shapes.geometric}
\usepackage{xcolor} % Required for custom/predefined colors
\usepackage[style=authoryear, backend=biber, maxcitenames=2]{biblatex}
\usepackage{titlesec}
\usepackage{fmtcount}
% Heading spacing
\titlespacing*{\section}{0pt}{1.5ex}{0.75ex}
\titlespacing*{\subsection}{0pt}{1.25ex}{0.6ex}
\titlespacing*{\subsubsection}{0pt}{1ex}{0.5ex}

\addbibresource{references.bib}
% Define light blue if you want a custom shade, or use builtin 'blue!50' or 'cyan!60'
\definecolor{customlightblue}{RGB}{70, 130, 180}
\makeatletter
\renewcommand{\@makechapterhead}[1]{%
	\vspace*{0pt}%
	{\parindent \z@ \centering \normalfont
		\ifnum \c@secnumdepth >\m@ne
		\vskip 0.5pt
		\Huge\bfseries Chapter \Numberstring{chapter}\par\nobreak
		
		\vskip 0.5pt
		\fi
		\interlinepenalty\@M
		\Huge\bfseries #1\par\nobreak
		\vskip 8pt
}}
\makeatother

\linespread{1.25}

% Style for definitions and examples (usually upright font)
\theoremstyle{definition}
% Define your environments, linking them to the 'section' counter
\newtheorem{definition}{Definition}[section] % Numbered by section (e.g., Definition 2.1.1)
\newtheorem{example}{Example}[section]       % Numbered by section (e.g., Example 2.1.1)

% Style for theorems (usually italicized font)
\theoremstyle{theorem}
% Define your environments, linking them to the 'section' counter
\newtheorem{theorem}{Theorem}[section]       % Numbered by section (e.g., Theorem 2.1.1)

% Remove page numbers
\pagestyle{empty}

% Adjust spacing between paragraphs
\setlength{\parskip}{0.3em} % Slightly reduced paragraph spacing

% Define a command for the "*" lines
\newcommand{\stars}{\par\noindent\textsuperscript{**********}\par}
\begin{document}
"""

EXACT_COVER_PAGE = r"""	\begin{center}
		\textbf{THE UNIVERSITY OF BAMENDA}\\
		\vspace{0.5cm} 
		\begin{minipage}[t]{0.4\textwidth}
			\centering
			\textbf{COLLEGE OF TECHNOLOGY}
			
		\end{minipage}
		% - University Seal/Logo goes here, between the two language blocks -
		\begin{minipage}[t]{0.08\textwidth} % Keep [t] alignment for the minipage
			\centering
			\begin{figure}[H]
				\centering 
			\includegraphics{images/logo.png}
			\end{figure} 
			% ================================================================
		\end{minipage}
		\hfill % Puts space between the logo minipage and the French minipage
		\begin{minipage}[t]{0.4\textwidth}
			\centering
			\textbf{DEPARTMENT OF COMPUTER ENGINEERING - DATA SCIENCE}
			
		\end{minipage}
	\end{center}
	
	\vspace{0.7cm} % Space between the heading block and the main title
	
	\noindent
	\begin{tikzpicture}
		\node[
		draw=customlightblue,
		line width=2.5pt,
		rounded corners=20pt,
		align=center,
		text width=\dimexpr\linewidth-20pt\relax, % Expands to match full page/text width
		inner sep=10pt % Inner padding
		] {
			\textbf{\large Evaluating Parameter-Efficient Fine-Tuning and Data Efficiency in Multilingual Transformer-Based Automatic Speech Recognition for the Kom Language}
		};
	\end{tikzpicture}
		
	\vspace{0.8cm} % Space below the main title
	
	\begin{onehalfspacing} % Set line spacing to 1.5
		\noindent A Dissertation submitted to the Department of Computer Engineering in the college of Technology of the University Of Bamenda in partial fulfillment of the requirements for the award of the degree of \textbf{MASTERS OF TECHNOLOGY in Computer Engineering - Data Science}.
	\end{onehalfspacing}
	
	\vspace{1cm} % Space below the dissertation text
	
	\begin{center}
		\textit{By} \\
		\vspace{0.1cm} % Small adjustment
		\textbf{\large MAYENMEIN TERENCE SAMA ALOAH JR} \\
		Registration N\textsuperscript{o}: \quad UBA25PP122
	\end{center}
	
	\vspace{0.8cm} % Space below the author's name
	
	\begin{center}
		\textit{SUPERVISOR} \\
		\vspace{0.1cm} % Small adjustment
		\textbf{\large MR. NJONGWE JAMES AGHEYIH }
	\end{center}
	
	\vspace{0.5cm}  
	
	\begin{center}
		\textbf{\large JUNE, 2025}
	\end{center}
"""


def _resolve_safe_tex_path(filename: str | None, output_dir: Path = OUTPUT_DIR) -> Path:
    """Ensures `filename` stays strictly inside `output_dir` and ends with `.tex`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_name = (filename or "").strip() or DEFAULT_TARGET_FILENAME
    cleaned = Path(raw_name)
    if cleaned.is_absolute() or ".." in cleaned.parts:
        raise ValueError(f"Invalid output filename '{raw_name}': path traversal is not allowed.")
    if cleaned.suffix.lower() != ".tex":
        cleaned = cleaned.with_suffix(".tex")
    resolved = (output_dir / cleaned).resolve()
    if output_dir not in resolved.parents and resolved != output_dir:
        raise ValueError(f"Resolved path '{resolved}' escapes output directory '{output_dir}'.")
    return resolved


def _ensure_bib_and_assets_in_output_dir(output_dir: Path) -> None:
    """Copies `references.bib` and `images/` assets into `output_dir` when available."""
    search_roots = [Path(__file__).resolve().parent, BASE_DIR]

    dest_bib = output_dir / "references.bib"
    if not dest_bib.exists():
        for root in search_roots:
            candidate_bib = root / "references.bib"
            if candidate_bib.exists():
                shutil.copy2(candidate_bib, dest_bib)
                break

    dest_images = output_dir / "images"
    if not dest_images.exists():
        for root in search_roots:
            candidate_images = root / "images"
            if candidate_images.is_dir():
                shutil.copytree(candidate_images, dest_images, dirs_exist_ok=True)
                break


def _insert_before_bibliography_or_end(existing_text: str, block_to_add: str) -> str:
    """Inserts new chapter/section LaTeX before `\\printbibliography` or `\\end{document}`."""
    bib_pattern = re.compile(r"^(\s*\\printbibliography\b.*)$", flags=re.MULTILINE)
    if bib_pattern.search(existing_text):
        return bib_pattern.sub(lambda m: f"\n{block_to_add}\n{m.group(1)}", existing_text, count=1)

    end_doc_pattern = re.compile(r"^(\s*\\end\{document\}\s*)$", flags=re.MULTILINE)
    if end_doc_pattern.search(existing_text):
        return end_doc_pattern.sub(lambda m: f"\n{block_to_add}\n{m.group(1)}", existing_text, count=1)

    sep = "\n\n" if not existing_text.endswith("\n\n") else ""
    return f"{existing_text}{sep}{block_to_add}"


def append_latex_section(
    latex_content: str,
    target_filename: str = DEFAULT_TARGET_FILENAME,
    section_key: str | None = None,
    include_preamble_and_cover: bool = True,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    """Appends LaTeX section content to `.tex` without reading prior file contents back to LLM.

    Whenever creating a new file (or overwriting), `include_preamble_and_cover=True` writes
    `EXACT_THESIS_PREAMBLE` and `EXACT_COVER_PAGE` automatically.
    """
    content = (latex_content or "").strip()
    if not content:
        return {"status": "error", "message": "latex_content was empty; nothing written."}

    try:
        tex_path = _resolve_safe_tex_path(target_filename)
        _ensure_bib_and_assets_in_output_dir(tex_path.parent)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    marker = f"% SECTION_KEY: {section_key.strip()}" if section_key and section_key.strip() else None

    if tex_path.exists() and not overwrite_existing:
        existing_text = tex_path.read_text(encoding="utf-8")
        if marker and marker in existing_text:
            return {
                "status": "skipped_duplicate",
                "file": tex_path.name,
                "file_path": str(tex_path),
                "section_key": section_key,
            }

        block_to_add = f"{marker}\n{content}\n" if marker else f"{content}\n"
        updated_text = _insert_before_bibliography_or_end(existing_text, block_to_add)
        tex_path.write_text(updated_text, encoding="utf-8")
        return {
            "status": "appended",
            "file": tex_path.name,
            "file_path": str(tex_path),
            "section_key": section_key,
            "bytes_written": len(block_to_add.encode("utf-8")),
        }

    header = f"{marker}\n" if marker else ""
    if include_preamble_and_cover and r"\begin{document}" not in content:
        final_text = (
            f"{EXACT_THESIS_PREAMBLE}\n"
            f"{EXACT_COVER_PAGE}\n\n"
            f"{header}{content}\n\n"
            r"\printbibliography" + "\n\n"
            r"\end{document}" + "\n"
        )
    else:
        final_text = f"{header}{content}\n"

    tex_path.write_text(final_text, encoding="utf-8")
    return {
        "status": "created",
        "file": tex_path.name,
        "file_path": str(tex_path),
        "section_key": section_key,
        "bytes_written": len(final_text.encode("utf-8")),
    }


def extract_latex_errors(
    target_filename: str = DEFAULT_TARGET_FILENAME,
    max_errors: int = 15,
) -> dict[str, Any]:
    """Extracts compact error lines from `.log` without returning `.tex` contents."""
    try:
        tex_path = _resolve_safe_tex_path(target_filename)
    except ValueError as exc:
        return {"status": "error", "message": str(exc), "errors": []}

    log_path = tex_path.with_suffix(".log")
    if not log_path.exists():
        return {
            "status": "no_log",
            "tex_file": tex_path.name,
            "errors": [],
        }

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    collected: list[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("!") or "LaTeX Error:" in stripped or re.match(r"^l\.\d+", stripped):
            entry = stripped
            if stripped.startswith("!") and idx + 1 < len(lines) and lines[idx + 1].strip().startswith("l."):
                entry = f"{stripped} ({lines[idx + 1].strip()})"
            if entry not in collected:
                collected.append(entry)
        if len(collected) >= max_errors:
            break

    return {
        "status": "collected",
        "tex_file": tex_path.name,
        "error_count": len(collected),
        "errors": collected,
    }


def _run_compile_passes(tex_path: Path, engine: str = "pdflatex", run_biber: bool = True) -> tuple[bool, list[str]]:
    """Runs pdflatex (and biber when present) against `tex_path` in `OUTPUT_DIR`."""
    cmd_tex = [
        engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={OUTPUT_DIR}",
        str(tex_path),
    ]
    proc = subprocess.run(cmd_tex, cwd=str(OUTPUT_DIR), capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        err_info = extract_latex_errors(tex_path.name)
        return False, err_info.get("errors", [])

    if run_biber and shutil.which("biber"):
        subprocess.run(
            ["biber", tex_path.stem],
            cwd=str(OUTPUT_DIR),
            capture_output=True,
            text=True,
            timeout=60,
        )
        for _ in range(2):
            proc2 = subprocess.run(cmd_tex, cwd=str(OUTPUT_DIR), capture_output=True, text=True, timeout=60)
            if proc2.returncode != 0:
                err_info = extract_latex_errors(tex_path.name)
                return False, err_info.get("errors", [])

    pdf_path = tex_path.with_suffix(".pdf")
    return pdf_path.exists(), []


def compile_latex_file(
    target_filename: str = DEFAULT_TARGET_FILENAME,
    engine: str = "pdflatex",
    run_biber: bool = True,
) -> dict[str, Any]:
    """Compiles a `.tex` file in `OUTPUT_DIR` and returns compact compilation status."""
    allowed_engines = {"pdflatex", "xelatex", "lualatex"}
    if engine not in allowed_engines:
        return {"status": "error", "message": f"Unsupported engine '{engine}'."}

    try:
        tex_path = _resolve_safe_tex_path(target_filename)
        _ensure_bib_and_assets_in_output_dir(tex_path.parent)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}

    if not tex_path.exists():
        return {"status": "error", "message": f"File '{tex_path.name}' not found."}

    try:
        ok, errors = _run_compile_passes(tex_path, engine=engine, run_biber=run_biber)
    except FileNotFoundError:
        return {"status": "error", "tex_file": tex_path.name, "message": f"Engine '{engine}' not on PATH."}
    except subprocess.TimeoutExpired:
        return {"status": "error", "tex_file": tex_path.name, "message": "LaTeX compilation timed out."}

    pdf_path = tex_path.with_suffix(".pdf")
    return {
        "status": "success" if ok else "compile_error",
        "tex_file": tex_path.name,
        "pdf_file": pdf_path.name if pdf_path.exists() else None,
        "errors": errors,
    }


def apply_single_latex_fix(
    fixed_latex_block: str,
    section_key: str,
    target_filename: str = DEFAULT_TARGET_FILENAME,
    engine: str = "pdflatex",
    run_biber: bool = True,
) -> dict[str, Any]:
    """Applies at most ONE correction write + recompile for `(target_filename, section_key)`."""
    attempt_key = f"{target_filename.strip()}::{section_key.strip()}"
    if attempt_key in _CORRECTION_ATTEMPTS_USED:
        latest_errors = extract_latex_errors(target_filename)
        return {
            "status": "manual_review_required",
            "fix_applied": False,
            "compile_success": False,
            "message": (
                f"Single correction attempt already used for '{attempt_key}'. "
                "No further retries allowed; manual review required."
            ),
            "errors": latest_errors.get("errors", []),
        }

    _CORRECTION_ATTEMPTS_USED.add(attempt_key)

    try:
        tex_path = _resolve_safe_tex_path(target_filename)
    except ValueError as exc:
        return {"status": "error", "fix_applied": False, "compile_success": False, "message": str(exc), "errors": []}

    if not tex_path.exists():
        return {"status": "error", "fix_applied": False, "compile_success": False, "message": "Target .tex file missing.", "errors": []}

    cleaned_fix = (fixed_latex_block or "").strip()
    if not cleaned_fix:
        return {"status": "error", "fix_applied": False, "compile_success": False, "message": "fixed_latex_block was empty.", "errors": []}

    existing_text = tex_path.read_text(encoding="utf-8")
    marker = f"% SECTION_KEY: {section_key.strip()}"
    replacement_block = f"{marker}\n{cleaned_fix}\n"
    replacement_text = replacement_block.rstrip()

    block_pattern = re.compile(
        rf"{re.escape(marker)}\n.*?(?=\n% SECTION_KEY: |\n\\printbibliography|\n\\end\{{document\}}|\Z)",
        flags=re.DOTALL,
    )
    if block_pattern.search(existing_text):
        updated_text = block_pattern.sub(lambda _: replacement_text, existing_text, count=1)
    else:
        updated_text = _insert_before_bibliography_or_end(existing_text, replacement_block)

    tex_path.write_text(updated_text, encoding="utf-8")

    try:
        ok, errors = _run_compile_passes(tex_path, engine=engine, run_biber=run_biber)
    except Exception as exc:
        return {
            "status": "manual_review_required",
            "fix_applied": True,
            "compile_success": False,
            "message": f"Re-compile error: {exc}. Manual review required.",
            "errors": [],
        }

    pdf_path = tex_path.with_suffix(".pdf")
    if ok:
        return {
            "status": "fixed_and_compiled",
            "fix_applied": True,
            "compile_success": True,
            "tex_file": tex_path.name,
            "pdf_file": pdf_path.name if pdf_path.exists() else None,
            "errors": [],
        }

    return {
        "status": "manual_review_required",
        "fix_applied": True,
        "compile_success": False,
        "tex_file": tex_path.name,
        "errors": errors,
        "message": "Single correction attempt did not resolve compilation errors; manual review required.",
    }
