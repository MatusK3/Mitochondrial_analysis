# Mitochondrial texture analysis

Diplomová práca


Ciel:
Implementovať nástorj, ktorý na základe poskytnutého datasetu mikroskopický snímkov mitochondií dokáže určiť najvystížnejšie textúrne príznaky pre daný problém. Zameriavame sa na proces degradácie zdravých mitochondrií. Za pomoci zvolených textúrnych príznakov potom tento nástroj má dokázať kvantifikovať akýkoľvek stav mitochondrii z daného rozsahu degradácie.

Pouzite kniznice
mirp: https://github.com/oncoray/mirp
featurewiz: https://github.com/AutoViML/featurewiz
pytorch: https://github.com/pytorch/pytorch

Plán:
* ✅ Inštalácia knižníc potrebných na extrakciu textúrnych príznakov z 2D dát
     - ✅ mirp, featurewiz
     - ✅ prieskum užitočný metód, ich vlastností a parametrov
* 🛠️ Zhotovenie datasetu (v priebehu)
     - ⏳ snímky mitochnodrii z ďalších prostredí
     - ✅ anotácia segmentácii zatial dostopnych vzoriek 
* 🛠️ Vhodné predspracovanie dát
     - ✅ Načítavanie datasetu
     - ⏳ automaticka segmentacia z fluoroescentných snimkov
     - ✅ Selekcia ROI, indicidualne bunky zo segmentacie (momentalne ručne anotované)
     - ✅ idividualne po bunke: normalizacia na rozsah 0..1, odstranenie šumu 
* 🛠️ Extrakcia príznakov z datasetu
     - ✅ diskretizacia snimokv do 32 fixných binov
     - ✅ Základná sada prvo-rádovýc príznakov a textúrnych príznakov
     - ✅ LBP filter
     - 🛠️ Rozšírit mnozstvo priznakov, pridanie filtrov
* 🛠️ Metóda na určenie najvystížnejších príznakov z datasetu
     - ✅ prvotne odfiltorvanie redudantných priznakov 
     - ✅ brute force selekcia najlepsich n priznakov podla presnosti s knn klasifikatorom
     - ⏳ lasso selekcia priznakov
     - 🛠️ prieskum dlaších pristupov k selekcii priznakov
* 🛠️ Kvantifikácia vzoriek pomocou zvolených prźnakov
     - ✅ knn klasifikator podla zvenych priznakov
     - ⏳ ďalšie klasifikatory
* ⏳ Vyhodnotenie dosiahnutých výsledkov
* 🛠️ Spísanie dokumentu záverečnej práce