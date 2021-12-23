#!/usr/bin/env python
# coding: utf-8

# # Klasterska analiza

# >U ovoj svesci ćemo uraditi primer klasterske analize sa većim brojem indikatora. Pomoću faktorske analize ćemo proceniti koliko je faktora za ovaj skup podataka zaista važno i pokušaćemo da ih opišemo. Primer je urađen za klasifikaciju država prema stepenu ekonomskog razvoja.

# Pravljenje rang-lista je jednostavno kada objekte treba poređati po veličini jednog parametra. Mi, međutim, često pravimo rang-liste za vrlo složene i apstraktne konstrukte kao što su konkurentnost privrede, kvalitet života, inovativnost itd. Da bi takve liste bile nešto više od subjektivnog rangiranja potrebno je da izaberemo najvažnije karakteristike, da procenimo koliko su važne, da li su možda međusobno previše zavisne i da smislimo odgovarajuću metriku.

# Svetska banka klasifikuje države i ekonomije prema stepenu ekonomskog razvoja u [četiri grupe](https://blogs.worldbank.org/opendata/new-country-classifications-income-level-2019-2020): visoku, gornju-srednju, donju-srednju i nisku. Za to koristi samo jedan parametar: bruto nacionalni proizvod (BNP, eng. GNP - _Gross national product_) po glavi stanovnika. Nesumnjivo, prosečni proizvod stanovnika određene teritorije jeste važan pokazatelj njene razvijenosti. Ipak, postoje velike razlike među državama sa sličnim BNP po glavi stanovnika. Na taj način u istu grupu stavlja npr. Norvešku i Saudijsku Arabiju ili Kinu i Angolu. Jasno nam je da samo jedan parametar ne može da bude dobra osnova za dobru klasifikaciju prema ekonomskom razvoju. Takođe poređenje ekonomskog razvoja sa indikatorima iz drugih oblasti neće dati značajne rezultate jer su grupe previše grubo određene. Zamislite da poredimo postignuće na PISA testu za različite grupe zemalja gde su u istoj grupi npr. Finska i Oman. Varijansa bi bila prevelika da bi poređenje imalo smisla. Potrebno nam je još parametara.

# Na sajtu Svetske banke postoji [kolekcija različitih indikatora](https://data.worldbank.org/indicator) o stepenu razvoja svih država i ekonomija koji se prikupljaju iz različitih izvora i po različitoj metodologiji. Sistematizacija ovih podataka nije jednostavan posao jer ne postoje svi podaci za sve zemlje i nema istraživanja za svaku godinu. Zato je potrebno napraviti odabir indikatora i kriterijum po kom ih povezujemo.

# ## Podaci
# 
# Za ovu vežbu iskoristiti fajl "countries data.csv" sa 29 indikatora koje je sakupio i priredio [Zedric Cheung](https://towardsdatascience.com/factor-analysis-cluster-analysis-on-countries-classification-1bdb3d8aa096). Ovi indikatori opisuju ne samo ekonomski već i društveni razvoj. U nedostatku podataka za aktuelnu godinu, uzeti su najnoviji dostupni podaci. Da bismo izbegli moguću pristrasnost zbog nesrazmernih vrednosti za određene indikatore, kao što su npr. BNP ili broj stanovnika, izabrani su samo relativni indikatori i indikatori rasta koji se uglavnom izražavaju u procentima.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('data/df_final.csv',index_col=0)


# In[3]:


df.head()


# Tabela je, evidentno, napravljena tako da su faktori prikazani po redovima, a države po kolonama. To nam nije zgodno za funkcije poput `corr()` koje koriste kolone kao argumente. Zato ćemo pomoću funkcije `T` transponovati _DataFrame_, tj. okrenuti tabelu tako da zamenimo redove i kolone.

# In[4]:


df=df.T


# In[5]:


df.head()


# Sada su države po vrstama, a indikatori po kolonama pa možemo da vidimo spisak svih odabranih indikatora.

# In[6]:


indicators=list(df.columns)
countries=list(df.index)


# In[7]:


indicators


# ## Faktorska analiza

# Osnovna namena faktorske analize jeste da opiše međusobne veze nekoliko promenljivih koristeći manji broj dimenzija koje ne možemo direktno da posmatramo i merimo. Ove dimenzije nazivamo faktorima.

# To što smo uzeli 29 indikatora ne znači da su svi nezavisni. Iza njih se verovatno krije svega nekoliko faktora koji najvećim delom objašnjavaju vrednosti indikatora. Faktorska analiza može da nam pokaže koji su to faktori najvažniji i šta ih čini.

# Bez namere da ovde ulazimo u detalje faktorske analize, iskoristićemo ovaj model za redukciju broja indikatora kojima opisujemo ekonomiju jedne zemlje. Ovaj postupak se zove smanjenje dimenzionalnosti (eng. _dimensionality reduction_).

# Faktorska analiza podrazumeva linearnu vezu promenljivih i zajedničkih faktora. Posmatrane promenljive se modeliraju kao linearne kombinacije faktora i statističkih grešaka. Ova pretpostavka se koristi za objašnjenje varijanse posmatrane promenljive i uočavanje faktora. Ukoliko postoji jaka nelinearna veza između promenljivih, onda faktorska analiza neće biti adekvatna.

# Faktorska analiza, takođe, pretpostavlja da promenljive mogu da budu međusobno zavisne. Na taj način promenljive mogu da se grupišu prema nekom zajedničkom svojstvu. U tom slučaju sve promenljive iz jedne grupe imaju međusobno visoke korelaciije dok su korelacije sa promenljivim iz drugih grupa niske. O svakoj grupi ovde možemo da razmišljamo kao o reprezentaciji istog osnovnog konstrukta ili faktora.

# ### Korelaciona matrica

# Pre nego što krenemo u dublju analizu, dobro je pogledati kako indikatori međusobno koreliraju. To može da nam pokaže korelaciona matrica.

# In[8]:


plt.figure(figsize=(10,8))
corrMatrix = df.corr()
sns.heatmap(corrMatrix)


# Odavde vidimo mnoge interesantne veze među promenljivim. Za neke od njih možemo da nađemo logično objašnjenje. Vidimo npr. pozitivnu korelaciju između dostupnosti električne energije i procenta ljudi koji koriste usluge osnovnog snabdevanja vodom ili negativnu korelaciju između udela ruralne populacije i udela populacije koja koristi internet. Ove dve veze nisu iznenađenje i za njih bismo mogli da nađemo prihvatljivo objašnjenje.
# 
# Za faktorsku analizu, međutim, nije neophodno da imamo objašnjenje uzročno-posledične veze između dve promenljive. Za analizu je važno da veza postoji, da može da se izmeri koliko je ona jaka i da na osnovu toga mogu da se grupišu. Iz zajedničkih svojstava tih grupa bi trebalo da prepoznamo faktore.

# Za kod koji sledi koristićemo biblioteku __factor_analyzer__ koju uglavnom nemamo u okruženju. Ako je već niste instalirali, možete to da uradite pomoću koda u sledećoj ćeliji. Naravno, uklonite # pre nego što pokrenete kod.

# In[9]:


# pip install factor_analyzer


# Da bismo videli koji je faktor koliko značajan izračunaćemo njihove jedinstvene vrednosti (_eigenvalues_). Pomoću funkcije `FactorAnalyzer()` ćemo konstruisati objekat __fa__ čije elemente dobijamo fitovanjem podataka datih u tabeli __df__ pomoću funkcije `fit()`. 
# 
# Funkcija `get_eingenvalues()` za objekat __fa__ vraća dva niza: originalne jedinstvene vrednosti i jedinstvene vrednosti za zajednički faktor. Nas će u ovom primeru interesovati samo prvi vraćeni niz __ev__.

# In[10]:


from factor_analyzer import FactorAnalyzer


# In[11]:


fa = FactorAnalyzer()
fa.fit(df)

ev, v = fa.get_eigenvalues()


# In[12]:


ev


# Dobili smo 29 jedinstvenih vrednosti za faktore koliko ima i indikatora. Ove vrednosti su već sortirane po veličini jedinstvenih vrednosti i vidimo da nemaju svi faktori isti značaj. Prvi faktor je svakako najznačajniji. Da bismo videli koliko ima značajnih faktora primenićemo Kajzerov kriterijum (_Kaiser criterion_). To znači da uzimamo samo one faktore sa jedinstvenom vrednošću većom ili jednakom 1. Pošto sedmi faktor ima jedinstvenu vrednost skoro 1 (0.976), možemo da uključimo i taj sedmi. Kad nacrtamo "_scree plot_" vidi se koje su jedinstvene vrednosti za koji faktor i gde bi trebalo da se zaustavimo.

# In[13]:


plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.hlines(1, 0, df.shape[1], colors='r')
plt.title('Scree Plot')
plt.xlabel('Faktori')
plt.ylabel('Jedinstvene vrednosti')
plt.grid()
plt.show()


# Uzeli smo prvih 7 faktora i sada nas interesuje za svaki indikator iz tabele __df__ koje su mu komponenente kad ga razložimo po faktorima. Opet ćemo pomoću funkcije `fit()` uraditi faktorsku analizu, samo što nas sada ne interesuju jedinstvene vrednosti nego komponente koje dobijamo funkcijom `loadings_`.
# 

# In[14]:


n=7
fa=FactorAnalyzer(n, rotation='varimax')
fa.fit(df)
loads=fa.loadings_
loads=pd.DataFrame(loads, index=df.columns)


# Težine faktora za svaki indikator, odnosno komponente indikatora po faktorima, nisu ništa drugo do koeficijenti korelacije između pojedinačnih indikatora i faktora. Kvadrat tog broja pokazuje koliki deo varijanse indikatora objašnjava svaki pojedinačni faktor.
# 
# U tabeli __loads__ vidimo težine faktora za svaku promenljivu. Dostupnost električne energije u ruralnim oblastima (__Access to electricity, rural__) je najvećim delom (0.923) posledica faktora 1, dok je npr. __Employment to population ratio, ages 15-24, total (%)__ najvećim delom posledica faktora 2. 

# In[15]:


loads


# Komponente pojedinačnih promenljivih možemo da prikažemo grafički pomoću funkcije `heatmap()`.

# In[16]:


plt.figure(figsize=(12,12))
sns.heatmap(loads, annot=True, cmap="coolwarm")


# Koliko dobro pomoću prvih sedam faktora možemo da opišemo indikatore, to kvantifikujemo pomoću udela varijanse za koju su odgovorni faktori. Za svaki faktor posebno možemo da vidimo koliki ima udeo ukupne varijanse. Kumulativni udeo nam govori koliko varijanse opisuju zajedno taj i svi prethodni faktori. Te podatke možemo da dobijemo pomoću funkcije `get_factor_variance()`. 

# In[17]:


fa_var = fa.get_factor_variance()
fa_var = pd.DataFrame(fa_var, index=['SS loadings', 'Proportion Var', 'Cumulative Var'])
fa_var


# Vidimo da je izbor prvih sedam faktora objasnio 71% ukupne varijanse svih 29 indikatora.

# ### Tumačenje faktora

# Iako su faktori apstraktni i ne možemo direktno da ih merimo, možemo da ih opišemo na osnovu veze sa pojedinačnim indikatorima. Npr. faktor 0 ima visoku korelaciju sa dostupnošću električne energije, interneta i pijaće vode. Možemo reći da ovaj faktor generalno opisuje dostupnost osnovnih komunalnih usluga. Slično možemo da prepoznamo i za ostale faktore.
# 
# Okvirno, sedam faktora ekonomskog razvoja na osnovu izabranih indikatora grubo možemo da opišemo kao:
# 
# 1. faktor — Dostupnost osnovnih komunalnih usluga
# 2. faktor — Zapošljivost mladih
# 3. faktor — Ukupni ekonomski rast
# 4. faktor — Industrijski razvoj
# 5. faktor — Zdravstvena situacija
# 6. faktor — Mogućnosti proizvodnje i trgovine proizvedenom robom
# 7. faktor — Razvoj stručnih usluga
# 
# Imajte u vidu da tumačenja podataka ne moraju da budu jedinstvena, da su veoma zavisna od konteksta i da su ponekad subjektivna. Zato je važno da naglasimo na osnovu čega dajemo takvo tumačenje.

# In[18]:


faktori=['Dostupnost osnovnih\n komunalnih usluga',
         'Zapošljivost mladih',
         'Ukupni ekonomski rast',
         'Industrijski razvoj',
         'Zdravstvena situacija',
         'Mogućnosti proizvodnje i\n trgovine proizvedenom robom',
         'Razvoj stručnih usluga']


# ## Klasterska analiza

# Vratimo se sada na svih 29 indikatora. Njih možemo predstaviti kao 29 dimenzija nekog hiper-prostora u kom je definisano rastojanje između bilo koje dve tačke. Ako krenemo _bottom-up_ i tražimo države koje su međusobno nabliže, počeće da se stvaraju klasteri država sa sličnim karakteristikama.

# ### Hijerarhijska klasterizacija

# Hijerarhijska klasterizacija je tip učenja bez nadzora koji grupiše slične tačke ili objekte u grupe koje nazivamo klasterima. Postoje dva podtipa hijerarhijske klasterizacije:
# - Aglomerativna hijerarhijska klasterizacija i
# - Hijerarhijska klasterizacija razdvajanjem.
# 
# Mi ćemo ovde prikazati primer korišćenja samo prvog od ova dva podtipa. Aglomerativna hijerarhijska klasterizacija ima _bottom-up_ pristup gde su svi objekti na početku zasebni klasteri koji se sukcesivno sparuju čineći sve veće klastere.

# #### Standardizacija

# Za uspešno uočavanje klastera potrebno je sve indikatore, bez obzira na raspon vrednosti i jedinice u kojima su dati, dovesti u ravnopravan položaj tako što im se vrednosti standardizuju. To činimo tako što originalne vrednosti indikatora zamenjujemo razlikom originalne i srednje vrednosti za taj indikator podeljenu standardnom devijacijom indikatora. To znači da su tipične standardizovane vrednosti uglavnom između -1 i +1.

# In[19]:


df_std=(df-df.mean())/df.std()


# Sa ovako transformisanom tabelom __df__ možemo da utvrdimo rastojanja između tačaka (država) koristeći podrazumevanu euklidsku distancu i Ward metod povezivanja klastera. Njihova klasterizacija može da se prikaže dijagramom koji nazivamo dendrogram zbog vizuelne sličnosti sa korenom drveta. Dendrogram prikazuje kako se klasteri spajaju ili razdvajaju na određenim rastojanjima. Skup tačaka može da ima različit broj klastera u zavisnosti od parametra kritičnog rastojanja koji je prikazan na y-osi. U krajnjoj liniji, izbor broja klastera je proizvoljan i vrlo često subjektivan.

# In[20]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[21]:


plt.subplots(figsize=(20,8))
dendrogram = sch.dendrogram(sch.linkage(df_std, method='ward'),leaf_rotation=90, 
                            leaf_font_size=8, color_threshold=12.5, labels=df.index)
plt.axhline(y=12.5, color='gray', lw=1, linestyle='--');


# Funkcija `dendrogram()` ne vraća podatke o klasterima jer još nismo zadali kriterijum koliki su ili koliko ih ima. Za grupisanje država u klastere nam je potrebna funkcija `AgglomerativeClustering()` u kojoj ćemo zadati da hoćemo tačno 12 klastera. To odgovara visini crne isprekidane linije na gornjem grafikonu. Svaki presek ove linije sa dendrogramom definiše po jedan klaster ispod linije. Funkcija `fit_predict()` vraća niz celih brojeva koji predstavljaju procenu pripadnosti tačaka različitim klasterima.

# In[22]:


# kreiranje objekta
hc = AgglomerativeClustering(n_clusters=12, affinity = 'euclidean', linkage = 'ward')
# fitovanje modela
y_hc = hc.fit_predict(df_std)


# Sada tabelama __df__ i njenoj standardizovanoj verziji (__df_std__) možemo da dodamo kolonu u kojoj će pisati kom klasteru pripada koja država. Nakon toga sortiramo tabele prema oznaci klastera tako da su nam na početku klasteri sa oznakom 0, pa onda 1 itd.

# In[23]:


df['cluster'] = y_hc
df.sort_values("cluster", inplace = True, ascending=True)

df_std['cluster'] = y_hc
df_std.sort_values("cluster", inplace = True, ascending=True)


# ### Klasteri država

# Sada pošto smo državama pridružili oznake klastera kojima pripadaju bilo bi dobro da vidimo koja je država u kom klasteru. Za tu namenu napravićemo privremeno tabelu __a__ koja sadrži samo imena država i oznake klastera i od nje napraviti tabelu __grupe__ gde su imena država odgovarajućeg klastera navedena po kolonama.

# In[24]:


a=pd.DataFrame({'country':df.index, 'cluster':df.cluster})


# In[25]:


mig=max(df.cluster.value_counts())
grupe=pd.DataFrame('',columns=df.cluster.unique(),index=range(mig))


# In[26]:


for i in range(12):
    _=a.country[a.cluster==i]
    grupe.iloc[0:len(_),i]=a.country[a.cluster==i]


# In[27]:


grupe


# Vidimo da su u prvom klasteru (sa oznakom 0) sve vrlo razvijene države poput Japana, Norveške, SAD itd. U drugom su mahom zemlje latinske Amerike. U drugoj uglavnom srednja i istočna Evropa, itd. Izgleda da klasterizacija ima smisla jer svaki klaster možemo da opišemo koristeći geografske i ekonomske atribute. Ovde se nećemo baviti opisom svakog pojedinačno klastera. To može da ostane za samostalni rad.

# ### Karakteristike klastera

# Nakon gruposanja 159 zemalja u 12 klastera, trebalo bi relativno lako da prepoznamo karakteristike svakog klastera i utvrdimo zašto su države baš na taj način grupisane. Za potrebe prikazivanja klastera i njihovih karakteristika, grupisaćemo vrednosti indikatora po faktorima računajući njihovu srednju vrednost. To će nam pomoći da vidimo po čemu se to faktori razlikuju. Grupisane vrednosti ćemo sačuvati u tabelama __df_cluster__ i __df_cluster_std__.

# In[28]:


df_cluster = df.groupby('cluster').mean()
df_cluster_std = df_std.groupby('cluster').mean()


# Ovako dobijene karakteristike faktora možemo grafički da prikažemo kao _heatmap_. Odavde možemo da vidimo koji indikatori najviše određuju pojedinačne klastere. Obratite pažnju da za ovaj grafički prikaz koristimo standardizovane vrednosti kako indikator sa najvećim rasponom vrednosti (_Secure Internet servers (per 1 million people)_) ne bi obesmislio prikaz za druge indikatore. (Probajte kako bi izgledao sledeći grafikon kada bismo koristili __df_cluster__ umesto __df_cluster_std__.)

# In[29]:


plt.figure(figsize=(20,10))
sns.heatmap(df_cluster_std, cmap="coolwarm", annot=True, fmt= '.2f', linewidths=.2)


# Ako nas interesuju konkretne vrednosti, onda nećemo koristiti standardizovanu tabelu __df_cluster_std__ već __df_cluster__. To možemo da prikažemo i kao _DataFrame_.

# In[30]:


df_cluster


# Odavde možete da vidite mnoštvo karakteristika klastera, npr. da zemlje četvrtog klastera imaju veoma malu dostupnost električne energije i da su im poljoprivreda i ribarstvo veoma važni za BDP. Ostavljamo vam za samostalni rad da prepoznate karakteristike pojedinačnih klastera. Još je bolje da te opise uparite sa geo-ekonomskim opisom država koje čine taj klaster.

# ## Smanjenje dimenzionalnosti

# Faktorska analiza se koristi kao jedan od načina za smanjenje broja dimenzija u skupu podataka sa kojim radimo. To je dobra praksa jer se tako odstranjuju nepotrebne promenljive i izbacuju one koje nisu nezavisne pa se zbog toga dobijaju pouzdaniji rezultati mašinskog učenja. Tehnički razlozi za smanjenje broja dimenzija su takođe veoma važni. Sa manjim skupom podataka se lakše i brže radi. Osim toga, previše promenljivih može da izazove _over-fitting_ efekat gde procene postaju lošije nego što bi bile da imamo manje promenljivih i jednostavniji model. Konačno, klasterska analiza u prostoru koji ima previše dimenzija može da postane neupotrebljiva zbog nedovoljno velikog uzorka za toliki broj dimenzija. Taj efekat se zove "prokletstvo multidimenzionalnosti".  

# Mi jesmo tražili rastojanja između tačaka u 29-odimenzionalnom prostoru za potrebe ove demonstracije, ali u opštem slučaju to nije dobra ideja. Smanjenje dimenzionalnosti (eng. _Dimensionality Reduction_) nam otvara mogućnost da radimo klasterizaciju sa manjim brojem promenljivih.

# Faktorska analiza može da predstavi faktore kao linearne kombinacije originalnih promenljivih pa da tako dobijemo numeričku vrednost faktora za svaku stavku u uzorku. Funkcija `transform()` koja je definisana za objekat koji smo kreirali za faktorsku analizu daje vrednosti faktora za svaku zemlju iz tabele __df__.
# 
# (Napomena: Pošto smo tabeli __df__ dodali _output_ kolonu __cluster__, ovde ćemo morati da je uklonimo jer nam smeta za funkciju `transform()`. To ćemo učiniti pomoću funkcije `drop()`.)

# In[31]:


dff=pd.DataFrame(fa.transform(df.drop('cluster',axis=1)),index=df.index,columns=faktori)


# Sada možemo da vidimo kolike su procenjene vrednosti (nemerljivih) faktora za koje od ranije imamo opise. Jasno je da su vrednosti faktora "Dostupnost osnovnih konumalnih usluga " značajno veće za Japan i Švedsku nego za Gvajanu i Belize.

# In[32]:


dff


# Pošto su nam prva dva faktora najznačajnija možemo da nacrtamo tačkasti dijagram gde će na x-osi biti prvi faktor, a na y drugi. Svaka tačka bi predstavila jednu zemlju, a njena boja klaster kom pripada. Ukoliko faktori imaju smisla trebalo bi da prepoznamo grupisanje tačaka iz istog klastera.

# In[33]:


prva=faktori[0]
druga=faktori[1]
plt.scatter(x=dff[prva],y=dff[druga],c=df['cluster'],cmap="Set1")
plt.xlabel(prva)
plt.ylabel(druga);


# Pošto tačke ne možemo da prikažemo u sedmodimenzionalnom prostoru, ostaje nam mogućnost da ih prikažemo za dve po dve dimenzije, tj. da nađemo sve parove faktora. Biblioteka __seaborn__ ima funkciju koja može u tome da nam pomogne: `pairplot()`.

# In[34]:


dff['cluster']=df['cluster']

sns.pairplot(dff,kind='scatter',hue='cluster',palette="Set1");


# Sa ovog grafikona vidimo da ne postoji jasna linearna veza između bilo koja dva faktora. To znači da su faktori prilično nezavisni i da smo napravili dobar izbor.

# ## Zaključak
# 
# Hijerarhijska klasterizacija je kompleksan algoritam koji postaje previše spor za veliki broj stavki. Ipak, za male i srednje velike skupove podataka predstavlja odličan izbor modela za mašinsko učenje bez nadzora. Paterni koje pronalazimo na ovaj način mogu da se objasne i da omoguće bolji uvid u podatke i pojave koje ih uzrokuju. Dodatna vrednost ovog modela je što može dobro da se vizuelizuje pa samim tima i da se rezultati bolje komuniciraju sa ciljnom grupom.
