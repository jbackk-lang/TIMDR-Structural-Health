# TIMDR-Structural-Health

<!-- TIMDR HUB LINKS -->
> **Ekosystem TIMDR:** [Knowledge Hub — dokumentacja i wyszukiwarka](https://jbackk-lang.github.io/TIMDR-AI-Knowledge-Base/) · [Panel badawczy](https://jbackk-lang.github.io/) · [GIA-TIMDR — źródła](https://github.com/jbackk-lang/GIA-TIMDR) · [Research AI — lokalne wyszukiwanie semantyczne](https://github.com/jbackk-lang/TIMDR-AI-Knowledge-Base). Publiczna wyszukiwarka Huba jest tekstowa; semantyczna wymaga lokalnego uruchomienia.
<!-- /TIMDR HUB LINKS -->

Monitorowanie konstrukcji — budynków, mostów, połączeń śrubowych — metodami zbudowanymi według drogowskazów TIMDR
(rama opisu sygnałów, repozytorium [GIA-TIMDR](https://github.com/jbackk-lang/GIA-TIMDR)). Jeden plik `timdr_shm.py`,
tylko numpy/scipy, testy wzorcowe zgodne 1:1 z wersjami walidowanymi.

## Co robi

| Funkcja | Droga TIMDR | Do czego |
|---|---|---|
| `anchor_frequencies` | kotwica modalna (gałąź K) z samokorektą ±4% | śledzenie częstotliwości własnych z drgań otoczenia; zmiana globalnej sztywności |
| `novelty` | detektor nowości kNN (mediana/MAD) | „czy konstrukcja zmieniła się względem stanu wzorcowego” |
| `lines_known_excitation` | linie przy kotwicy wymuszenia (2f, 3f) | nieliniowość kontaktu przy znanym wymuszeniu (luzowanie śruby) |
| `lines_two_modes` | linie przy dwóch modach (2fa, 2fb, fa+fb) | nieliniowość przy wymuszeniu losowym (zderzak, rysa „oddychająca”) |
| `regime` | cząsteczkowość P i gęstość zdarzeń D | czy zdarzenia są rzadkie i ostre (cząsteczka), czy zlane (fala) |
| `mode_shapes` | kotwica K + **faza** względem wzbudnika → kształt modu (G) | kształty modów na siatce czujników |
| `DiscontinuityBaseline` | skok M/S w przestrzeni: wzdłuż linii (gapped smoothing) i w poprzek (sąsiednie linie); **maksimum**, nie suma | lokalne uszkodzenie jako przerwa na czymś ciągłym (połączenie, przegub) |

```
python timdr_shm.py modes    rekord.npz --fs 825.8 --anchors 2.58,2.92,5.32 --decimate 32 --channels aBD11Az,aBD17Az
python timdr_shm.py lines    wibrometr.npz --fs 5000
python timdr_shm.py lines2   pietra.npz --fs 322.58 --band-a 45,62 --band-b 64,80
python timdr_shm.py regime   emisja.npz --fs 5e6 --band 100e3,900e3
```

Przerwa ciągłości (w Pythonie; `grid` = indeksy kanałów ułożone w linie × punkty o równym rozstawie):

```python
import timdr_shm as T
train = [T.mode_shapes(X_okno, wzbudnik_okno, fs, kotwice) for X_okno, wzbudnik_okno in okna_wzorcowe]
B = T.DiscontinuityBaseline(train, grid)
wynik = T.novelty(B.features(train), B.features([T.mode_shapes(X, wzbudnik, fs, kotwice)]))
za, zc = B.z_maps([...])   # gdzie: mapa |z| wzdłuż i w poprzek linii
```

## Dashboard

`run.bat` (Windows) albo `python dashboard.py` → http://127.0.0.1:8765. Tylko biblioteka standardowa + numpy/scipy,
serwer słucha wyłącznie na 127.0.0.1. Pierwsza zakładka „Jak widzi TIMDR” pokazuje inność metody na jednym sygnale krok po kroku: widmo (standard) →
typ sygnału (obwiednia, błyski, reżim) → pole czas × pasmo → mapa rezonansu i oczka sita przy kotwicy → co przepuszcza sito,
a co zwykła obwiednia (z obcą maszyną w tle) → zwinięcie w rurę; dla rzadkich i częstych uderzeń (w reżimie fali metoda
zmienia drogę). Pozostałe zakładki: most (kotwica modalna przed/po wzmocnieniu), śruba (linie nieliniowości
przy luzowaniu), uderzenia (reżim cząsteczka/pakiet/fala) — na danych syntetycznych; „Mój plik CSV” — analiza własnego
zapisu (kolumny = kanały); „Walidacja” — tabela wyników, także negatywnych.

## Walidacja (pre-rejestrowana w GIA-TIMDR, jedno uruchomienie, także wyniki negatywne)

| Obiekt | Wynik TIMDR | Odniesienie | Werdykt |
|---|---|---|---|
| **Most kolejowy KW51** (Leuven), przed/po wzmocnieniu | kotwica z 2 kanałów: AUC **1,00**; zgodność z analizą modalną autorów ≤ 1% (4/5 modów) | SSI autorów 1,00 (remis — sufit); AR 0,67 | **SUPPORTED** |
| **Rama 3-kondygnacyjna LANL**, zderzak (nieliniowość) | linie przy dwóch modach: ciężkość ρ **+0,98** | klasyczne SHM 0,97 | **SUPPORTED** (dane użyte trzeci raz — ujawnione) |
| **Połączenie śrubowe ORION-AE**, luzowanie 60 → 5 cNm | linie przy kotwicy: ρ −0,85 / −0,64 (inne kampanie); emisja akustyczna — reżim cząsteczki przy średnim luzie | klasyfikacja dokładnego poziomu: TIMDR 0,18, klasyka 0,04 — obie przy poziomie losowym | **SUPPORTED** (trend), klasyfikacja poziomu — nie |
| **Most Hell Bridge Test Arena**, 8 lokalnych uszkodzeń połączeń | sama kotwica + kształt bez fazy: AUC 0,65 | AR 0,79 | **NOT SUPPORTED** |
| **Most Hell Bridge** — przerwa ciągłości (K → G → M/S, v0.3) | uszkodzenia pionowe (5 stanów): AUC **0,93–0,94**, każde ≥ AR; średnio 0,80 | AR 0,80 (remis średnio; stężeń czujniki pionowe nie widzą) | częściowo: H2 **SUPPORTED**, H1 remis (dane użyte 4. raz — rozpoznawczo) |

Szczegóły: GIA-TIMDR `docs/geometry/RESULT_KW51_MODAL_ANCHOR_v0.1.md`, `RESULT_LANL_MODAL_ANCHOR_v0.3.md`,
`RESULT_ORION_BOLT_v0.1.md`, `RESULT_HBTA_MODAL_ANCHOR_v0.1.md`, `RESULT_HBTA_MODAL_CURVATURE_v0.2.md`,
`RESULT_HBTA_DISCONTINUITY_v0.3.md`.

## Granice

- **Działa:** zmiany globalnej sztywności (wzmocnienie, masa, sztywność), nieliniowość kontaktu przy wymuszeniu (zderzak,
  luzowanie śruby) — jako **trend**, nie dokładny poziom.
- **Lokalne uszkodzenia połączeń:** sama kotwica ich nie widzi (HBTA 0,65); dopiero kształt modu z fazą i szukanie przerwy
  ciągłości (maksimum skoku, nie suma) wykrywa uszkodzenia pionowe na poziomie ≥ AR. Wymaga siatki czujników o równym
  rozstawie i sygnału wzbudnika (albo kanału odniesienia) do fazy; rozdzielczość = rozstaw czujników.
- **Nie działa:** stężenia poprzeczne przy czujnikach tylko pionowych (HBTA DS5–7).
- Temperatura przesuwa częstotliwości (zwłaszcza poniżej 0 °C); próg alarmu trzeba ustalić na stanie wzorcowym danego
  obiektu w podobnych warunkach. Przy małej liczbie okien uczących próg 95% jest zbyt wysoki (HBTA).
- To nie jest certyfikowane narzędzie oceny bezpieczeństwa konstrukcji.

## Licencja

Copyright (c) 2026 Jacek Stanisław Kielich. Wszelkie prawa zastrzeżone. Dane testowe należą do ich autorów
(KW51: CC BY-NC-SA; ORION-AE: CC0; HBTA i LANL — patrz GIA-TIMDR).
