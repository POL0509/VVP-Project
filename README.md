# VVP-Project

From Beremi/vvp_fraktaly

# Vizualizace fraktálů

## Textový popis

Tento projekt se zaměřuje na vizualizaci fraktálů. Cílem je
implementovat algoritmy pro efektivní generování známých fraktálů, jako
jsou Mandelbrotova a Juliova množina (pro
$f\left(z\right)=z^{2}+c,c\in\mathbb{C}$), a vytvořit interaktivní
vizualizace (viz [Matplotlib
widgets](https://matplotlib.org/stable/gallery/widgets/index.html))
těchto fraktálů pomocí knihovny Matplotlib.

Výstupem projektu budou interaktivní vizualizace fraktálů, které
umožňují uživateli prozkoumávat různé části fraktálu a přizpůsobovat
parametry pro generování fraktálů ($c\in\mathbb{C}$ pro Juliovu
množinu).

## Funkcionality

-   Implementovat algoritmus pro efektivní generování Mandelbrotovy
    množiny pomocí knihovny NumPy
-   Implementovat algoritmus pro efektivní generování Juliovy množiny
    (pro $f\left(z\right)=z^{2}+c,c\in\mathbb{C}$) pomocí knihovny NumPy
-   Vytvořit funkci pro vizualizaci fraktálů pomocí knihovny Matplotlib,
    která zobrazuje fraktály pomocí barevného mapování podle iterací
    potřebných k dosažení určitého prahu
-   Implementovat interaktivní prvky vizualizace, které umožňují
    uživateli:
    -   přiblížit nebo oddálit fraktál
    -   měnit barevné schéma vykreslení počtu iterací do divergence
    -   přizpůsobovat parametry pro generování fraktálů (např. počet
        iterací, $c$)
