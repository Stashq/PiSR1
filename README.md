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

## Problemy opisywane w literaturze

- **Cold-start** - problem związany z wchodzeniem do systemu nowych użytkowników i obiektów, o których nic nie wiemy. Często stosowane są pytania do użytkowników przed dołączeniem do systemu lub pobieranie o nich informacji z innych źródeł (media społecznościowe, ect.)
- **Data sparsity** - macierz reprezentująca relacje użytkowników i przedmiotów zazwyczaj będzie rzadko wypełniona liczbami. Aby sobie z tym radzić można skorzystać z faktoryzacji macierzy.
- **Accuracy** - poprawa dokładności systemu jest trudnym zadaniem szczególnie gdy dane są rzadkie. Dodatkowo dochodzi problem z walidacją rozwiązania, część rozwiązań systemu nie może być porównana z rzeczywistym wynikiem, gdyż nie jesteśmy w stanie sprawdzić jak użytkownik zareaguje na kontent, który mu proponujemy (a takich przypadków może być bardzo dużo).
- **Scalability** - skalowalność związana jest z liczbą użytkowników i obiektów, dla których system ma działać. System zaprojektowany do polecania kilku obiektów setkom ludzi nie będzie działał przy kilku tysiącach przedmiotów i kilku milionach użytkowników, mimo iż proporcja między nimi zostanie zachowana. 
- **Diversity** - różnorodność to cecha porządana dla systemów rekomendujących. Bywa, że mają tendencję do faworyzowania pewnych obiektów. Literatura wskazuje takie rozwiązania jak _K-Furthest Neighbors_ (odwrotność KNN) lub znajdowywanie takich użytkowników, którzy mają uważani są za "ekspertów", mają "dobry smak" i podpowiadanie użytkownikom "normalnym" rzeczy, które lubią "eksperci".
- **Popularity bias** - występuje, gdy system rekomenduje obiekty z największą liczbą interakcji, bez jakiejkolwiek personalizacji.
- inne problemy takie jak: brak personalizacji, ochrona prywatności, redukcja szumów, integracja źródeł danych, brak nowości i adaptacja do preferencji użytkownika.

## Collaborative Filtering

Wyobraźmy sobie, że mamy macierz, której komórki reprezentują recenzje obiektów pozostawione przez użytkowników. W kolumnach umieszczone mamy recencje dla konkretnych obiektów, a w wierszach recencje konkretnych użytkowników. Wyróżniamy następujące rodzaje Collaborative Filteringu:

- **user-user** - porównujemy użytkowników (wiersze) i zwracamy dla użytkownika te obiekty, które wystąpiły u podobnych użytkowników z wysokim rankingiem. Działa świetnie gdy jest mało użytkowników (wierszy) i dużo obiektów (kolumn).
- **item-item** - porównujemy obiekty (kolumny) i obserwując oceny między nimi uzupełniamy ich wybrakowane oceny. Działa świetnie gdy jest mało obiektów (kolumn) i dużo użytkowników (wierszy). 
- **user-item** - wykorzystuje cechy obu poprzednich technik. Najprostsza z metod oparta jest na faktoryzacji macierzy, dzięki której otrzymujemy osadzenia opisujące jak bardzo dany obiekt zawiera daną cechę i jakimi obiektami interesuje się dany użytkownik. Najczęściej wykorzystywane metody z tej rodziny to:
  - **Singular Value Decomposition** - najpopularniejsza z tej rodziny metod. Przedmioty i użytkowników reprezentujemy w postaci wektorów tak, że po przemnożeniu otrzymujemy wysokość oceny. Metoda ta jest wymagająca obliczeniowo i słabo skalowanlna.
  - **Alternating Least Square** - metoda nadająca się do wykorzystania przy średniej wielkości danych.

## Content-based Filtering

_Collaborative Filtering_ cierpi na problem _cold-start_. Systemy nie mogą rekomendować coś czego nikt nie zarekomendował i nie są w stanie pokazać nowemu użytkownikowi trafioną propozycje nie znając jego gustu. Technika **Content-based Filtering** radzi sobie z tym po przez wprowadzenie do systemu wiedzy o obiektach i użytkownikach i wyliczaniu podobieństw między nimi. 

## Podejście hybrydowe

Modele hybrydowe oparte są na głębokich sieciach. Wykorzystują wiedzę zdobytą z _Collaborative_Filtering_, _Content-based Filtering_ oraz innych wybranych technik zależnie od implementacji. Ich zaletą jest to, że dzięki nieliniowości są w stanie oddać cięższe do wychwycenia niuanse dotyczące gustów. Ponadto są wstanie operować na danych z różnych dziedzin (np. obraz i dźwięk). Utworzenie ich wymaga jednak bardzo dużo obliczeń i eksperymentów z hiperparametrami. Techniki rekomendacji możemy łączyć na wiele sposobów. Do tych technik należą:

- **Weighted** - na podstawie wyników z kilku modeli produkuje się jedną rekomendację.
- **Switching** - system przełącza się między modelami.
- **Mixed** - system prezentuje wyniki z różnych modeli.
- **Feature combination** - cechy z wielu systemów przechodzą do następnego systemu rekomendacji.
- **Cascade** - wyniki rekomendacji przekazywane są z modelu do modelu.
- **Meta-level** - profil, którego nauczył się jeden model, jest używany jako dane wejściowe do innego.

## Techniki w uczeniu maszynowym

- **K-NN** - technika popularna dla Collaborative Filtering.
- **Clustering** - najpopularniejszy jest _K-means_.
- **Fuzzy logic** - uważane za komplementarne w stosunku do metod z rodziny Collaborative, często używane wraz z nimi.
- **Matrix manipulation** - należą do tej rodziny techniki takie jak: _Singular Value Decomposition_ (SVD), _Latent Dirichlet Allocation_ (LDA), _Principal Component Analysis_ (PCA), _Dimensionality Reduction_ oraz _similar matrix factorization_.
- inne, rzadziej wykorzystywane techniki: _Genetic Algorithms_, _Naive Bayes_, _Neural Networks_, _Notion of Experts_, _Statistical Modeling_, ect.

## Repository

### Research

You can find research notebooks at the `./notebooks` directory.
EDA is at `./notebooks/EDA` directory.

### Deserialization

You can find serialized models at the `./models` directory.

#### Matrix Factorization

```py
from pathlib import Path

import torch

from src.models.torch.mf import MatrixFactorization

model_path = Path('models/matrix_factorization.pt')
model: MatrixFactorization = torch.load(model_path)
model.eval()

with torch.no_grad():
    movies_ranking = model.predict(1)
    movies_ranking, scores = model.predict_scores(1)
    score = model.predict_score(1, 31)
```

#### Embedded Regression

```py
from pathlib import Path

import torch

from src.models.torch.embedded_regression import EmbeddedRegression

model_path = Path('models/embedded_regression.pt')
model: EmbeddedRegression = torch.load(model_path)
model.eval()

with torch.no_grad():
    movies_ranking = model.predict(1)
    movies_ranking, scores = model.predict_scores(1)
    score = model.predict_score(1, 31)
```
