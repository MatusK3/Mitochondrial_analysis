# vizual najlepších 2 príznakov

testoval som predspracovane a ne-predspracovane data. Predspracovanie zahrňa normalizacia hodnot na rozsah 0:1, po jednotlivych bunkach, a odstranenie šumu (skimage.restoration.denoise_nl_means() metodou, cely kod je v súbore img_preprocess.py)

Na vyhodnocovanie presnosti som používal knn klasfikátor, pre počet susedov k=3. Pre každú dvojicu príznakov som rátal priemernú presnosť z 10 pokusov, kde sa zakaždým náhodne zvolilo 30% dát ako validačných, na ktorých sa následne rátala presnosť. Na základe tejto priemernej presnosti sa potom vybralo 9 najlepšich dvjíc. tćhto 9 grafov so m vykreslil a uložil. Každy z grafov ma 2osi, zvolené priznaky, a nadpis s dosiahnutou presnosťou (neratál som testovaciu accuracy). Cely kod je v súbore classification.py

Dva priložene obrazky obsahuju opisane grafy.