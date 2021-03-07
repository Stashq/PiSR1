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
- **Diversity** - różnorodność to cecha porządana dla systemóœ rekomendujących. Bywa, że mają tendencję do faworyzowania pewnych obiektów. Literatura wskazuje takie rozwiązania jak _K-Furthest Neighbors_ (odwrotność KNN) lub znajdowywanie takich użytkowników, którzy mają uważani są za "ekspertów", mają "dobry smak" i podpowiadanie użytkownikom "normalnym" rzeczy, które lubią "eksperci".
- inne problemy takie jak: brak personalizacji, ochrona prywatności, redukcja szumów, integracja źródeł danych, brak nowości i adaptacja do preferencji użytkownika.

