"""export_view_page.py -- statyczna strona "Jak widzi TIMDR" (bez serwera) do GitHub Pages.

Bierze style i kod rysujacy z static/dashboard.html, wbudowuje dane dwoch przypadkow (view_case) i zapisuje jeden
samodzielny plik HTML. Uzycie: python export_view_page.py <plik_wyjsciowy.html>
"""
import json
import sys
from pathlib import Path

import dashboard as D

src = (Path(__file__).parent / "static" / "dashboard.html").read_text(encoding="utf-8")
style = src[src.index("<style>"):src.index("</style>") + len("</style>")]
js = src[src.index('<script>') + len('<script>'):src.index("async function showMost()")]
js = js.replace('await api("/api/view?case=" + kind)', "DATA[kind]")
data = {k: D.view_case(k) for k in D.VIEW_CASES}

page = f"""<!doctype html>
<html lang="pl" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jak widzi TIMDR</title>
<meta name="description" content="Ten sam sygnał oczami klasycznej analizy i oczami TIMDR: typ sygnału, pole, rezonans i sito, rura.">
{style}
<style>.top{{display:flex;gap:16px;flex-wrap:wrap;align-items:center;justify-content:space-between}} .top a{{color:var(--s1);text-decoration:none}}</style>
</head>
<body>
<header>
  <div class="top"><h1>Jak widzi TIMDR</h1><a href="index.html">← strona główna</a></div>
  <p class="sub">Jeden sygnał krok po kroku: jak patrzy na niego klasyczna analiza, a jak metoda zbudowana według drogowskazów TIMDR.
  Dane syntetyczne; liczby z prawdziwych testów w repozytoriach
  <a href="https://github.com/jbackk-lang/GIA-TIMDR">GIA-TIMDR</a> i
  <a href="https://github.com/jbackk-lang/TIMDR-Structural-Health">TIMDR-Structural-Health</a>.</p>
</header>
<main><section id="p-widok"></section></main>
<script>
"use strict";
const DATA = {json.dumps(data, ensure_ascii=False, separators=(",", ":"))};
{js}
showWidok();
</script>
</body>
</html>
"""
Path(sys.argv[1]).write_text(page, encoding="utf-8")
print("zapisano", sys.argv[1], len(page) // 1024, "kB")
