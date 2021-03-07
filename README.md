# PiSR1
Z1: Podstawowe zagadnienia systemów rekomendacyjnych

Tematyka najbliższego tygodnia:  
- collaborative filtering
- content-based filtering
- demographic recommendation
- hybrid approaches
- K-nearest neighbours classification
- porównanie miar podobieństwa

Wymagany czas: ok. 4h / os. pracy własnej:  
1. czytanie 1h
2. przykłady kodu, zapoznanie się z
istniejącymi projektami 1h
3. spotkanie/call grupowy - wymiana wiedzy
20 min
4. implementacja 1h
5. spotkanie / call grupowy - przygotowanie
prezentacji 40 minut


## Collaborative Filtering
Wyobraźmy sobie, że mamy macierz, której komórki reprezentują recenzje obiektów pozostawione przez użytkowników. W kolumnach umieszczone mamy recencje dla konkretnych obiektów, a w wierszach recencje konkretnych użytkowników. Wyróżniamy następujące rodzaje Collaborative Filteringu:
- **user-user** - porównujemy użytkowników (wiersze) i zwracamy dla użytkownika te obiekty, które wystąpiły u podobnych użytkowników z wysokim rankingiem. Działa świetnie gdy jest mało użytkowników (wierszy) i dużo obiektów (kolumn).
- **item-item** - porównujemy obiekty (kolumny) i obserwując oceny między nimi uzupełniamy ich wybrakowane oceny. Działa świetnie gdy jest mało obiektów (kolumn) i dużo użytkowników (wierszy). 
- **user-item** - wykorzystuje cechy obu poprzednich technik. Najprostsza z metod oparta jest na faktoryzacji macierzy, dzięki której otrzymujemy osadzenia opisujące jak bardzo dany obiekt zawiera daną cechę i jakimi obiektami interesuje się dany użytkownik. Najczęściej wykorzystywane metody z tej rodziny to:
    - **Singular Value Decomposition** - najpopularniejsza z tej rodziny metod. Przedmioty i użytkowników reprezentujemy w postaci wektorów tak, że po przemnożeniu otrzymujemy wysokość oceny. Metoda ta jest wymagająca obliczeniowo i słabo skalowanlna.
    - **Alternating Least Square** - metoda nadająca się do wykorzystania przy średniej wielkości danych.

## Problemy opisywane w literaturze

- **Cold-start** - problem związany z wchodzeniem do systemu nowych użytkowników i obiektów, o których nic nie wiemy. Często stosowane są pytania do użytkowników przed dołączeniem do systemu lub pobieranie o nich informacji z innych źródeł (media społecznościowe, ect.)
- **Data sparsity** - macierz reprezentująca relacje użytkowników i przedmiotów zazwyczaj będzie rzadko wypełniona liczbami. Aby sobie z tym radzić można skorzystać z faktoryzacji macierzy.
- **Accuracy** - poprawa dokładności systemu jest trudnym zadaniem szczególnie gdy dane są rzadkie. Dodatkowo dochodzi problem z walidacją rozwiązania, część rozwiązań systemu nie może być porównana z rzeczywistym wynikiem, gdyż nie jesteśmy w stanie sprawdzić jak użytkownik zareaguje na kontent, który mu proponujemy (a takich przypadków może być bardzo dużo).
- **Scalability** - skalowalność związana jest z liczbą użytkowników i obiektów, dla których system ma działać. System zaprojektowany do polecania kilku obiektów setkom ludzi nie będzie działał przy kilku tysiącach przedmiotów i kilku milionach użytkowników, mimo iż proporcja między nimi zostanie zachowana. 
- **Diversity** - różnorodność to cecha porządana dla systemów rekomendujących. Bywa, że mają tendencję do faworyzowania pewnych obiektów. Literatura wskazuje takie rozwiązania jak _K-Furthest Neighbors_ (odwrotność KNN) lub znajdowywanie takich użytkowników, którzy mają uważani są za "ekspertów", mają "dobry smak" i podpowiadanie użytkownikom "normalnym" rzeczy, które lubią "eksperci".
- **Popularity bias** - występuje, gdy system rekomenduje obiekty z największą liczbą interakcji, bez jakiejkolwiek personalizacji.
- inne problemy takie jak: brak personalizacji, ochrona prywatności, redukcja szumów, integracja źródeł danych, brak nowości i adaptacja do preferencji użytkownika.

## Techniki w uczeniu maszynowym
- **K-NN** - technika popularna dla Collaborative Filtering.
- **Clustering** - najpopularniejszy jest _K-means_. 
- **Fuzzy logic** - uważane za komplementarne w stosunku do metod z rodziny Collaborative, często używane wraz z nimi.
- **Matrix manipulation** - należą do tej rodziny techniki takie jak: _Singular Value Decomposition_ (SVD), _Latent Dirichlet Allocation_ (LDA), _Principal Component Analysis_ (PCA), _Dimensionality Reduction_ oraz _similar matrix factorization_.
- inne, rzadziej wykorzystywane techniki: _Genetic Algorithms_, _Naive Bayes_, _Neural Networks_, _Notion of Experts_, _Statistical Modeling_, ect.

