import pandas as pd
import ast
import numpy as np
from collections import defaultdict, deque

# Daten laden
df_task2_annotation = pd.read_pickle(
    "./annotation/analysis/survey/export/task_2_annotation_survey.pkl"
)

df_task2_annotation.head()
df_task2_annotation.columns
df_task2_annotation['feature_four'].head(10)

# Feature definieren (kann angepasst werden)
focus_feature = "feature_four"  # oder "feature_one", "feature_five", etc.

# Funktion zum Zählen der Events in einer Annotation
def count_events(val):
    """Zählt die Anzahl der Events in einem Feature"""
    if not val or (isinstance(val, float) and pd.isna(val)):
        return 0
    
    if isinstance(val, str):
        if val.strip() == "*":
            return 0
        try:
            val = ast.literal_eval(val)
        except Exception:
            return 0
    
    # Prüfe auf Set (bereits verarbeitet)
    if isinstance(val, set):
        return len(val)
    
    # Prüfe auf Liste
    if isinstance(val, list):
        return len(val)
    
    return 0

def calculate_dag_metrics(triples):
    """
    Berechnet DAG-Metriken:
    - Anzahl der Kanten (= Anzahl der Triples/Relationen)
    - Maximale Pfadtiefe (längste Kette von Wurzel zu Blatt)
    """
    if not triples or (isinstance(triples, float) and pd.isna(triples)):
        return 0, 0
    
    if isinstance(triples, str):
        if triples.strip() == "*":
            return 0, 0
        try:
            triples = ast.literal_eval(triples)
        except Exception:
            return 0, 0
    
    if isinstance(triples, set):
        triples = list(triples)
    
    if not isinstance(triples, list) or len(triples) == 0:
        return 0, 0
    
    # Anzahl der Kanten
    num_edges = len(triples)
    
    # Baue einen gerichteten Graphen
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    all_nodes = set()
    
    for triple in triples:
        if len(triple) >= 3:
            source = triple[0]
            target = triple[2]
            graph[source].append(target)
            in_degree[target] += 1
            all_nodes.add(source)
            all_nodes.add(target)
            # Sicherstellen, dass source auch in in_degree ist
            if source not in in_degree:
                in_degree[source] = 0
    
    if len(all_nodes) == 0:
        return 0, 0
    
    # Finde Wurzelknoten (Knoten ohne eingehende Kanten)
    root_nodes = [node for node in all_nodes if in_degree[node] == 0]
    
    if not root_nodes:
        # Falls Zyklus oder alle Knoten Eingänge haben, nimm längsten Pfad von jedem Knoten
        root_nodes = list(all_nodes)
    
    # BFS um längsten Pfad von Wurzelknoten zu finden
    def get_max_depth_from_node(start_node):
        queue = deque([(start_node, 0)])
        visited = {start_node}
        max_depth = 0
        
        while queue:
            node, depth = queue.popleft()
            max_depth = max(max_depth, depth)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        return max_depth
    
    # Maximale Pfadtiefe von allen Wurzelknoten
    max_depth = max(get_max_depth_from_node(root) for root in root_nodes)
    
    return num_edges, max_depth

# Event-Anzahl für jede Zeile berechnen
df_task2_annotation['event_count'] = df_task2_annotation[focus_feature].apply(count_events)

# DAG-Kettenlängen berechnen (aus triples)
df_task2_annotation[['num_edges', 'max_depth']] = df_task2_annotation['triples'].apply(
    lambda x: pd.Series(calculate_dag_metrics(x))
)

# Filtere Daten für Kettenlängen-Statistiken (nur nicht-null Ketten)
df_with_chains = df_task2_annotation[df_task2_annotation['max_depth'] > 0].copy()

# Statistiken pro Annotator berechnen
annotator_stats = df_task2_annotation.groupby('annotator').agg({
    'event_count': ['mean', 'median', 'std', 'min', 'max', 'sum', 'count']
}).round(2)

# Kettenlängen-Statistiken nur für Annotationen mit Ketten
if len(df_with_chains) > 0:
    chain_stats = df_with_chains.groupby('annotator').agg({
        'num_edges': ['mean', 'median', 'std', 'min', 'max'],
        'max_depth': ['mean', 'median', 'std', 'min', 'max']
    }).round(2)
    
    # Kombiniere die Statistiken
    annotator_stats = pd.concat([annotator_stats, chain_stats], axis=1)
else:
    # Falls keine Ketten vorhanden, füge leere Spalten hinzu
    for col in ['num_edges', 'max_depth']:
        for stat in ['mean', 'median', 'std', 'min', 'max']:
            annotator_stats[(col, stat)] = np.nan

# Spalten umbenennen für bessere Lesbarkeit
annotator_stats.columns = [
    'Events_Durchschnitt', 'Events_Median', 'Events_Std', 'Events_Min', 'Events_Max', 'Events_Gesamt', 'Anzahl_Annotationen',
    'AnzahlKanten_Durchschnitt', 'AnzahlKanten_Median', 'AnzahlKanten_Std', 'AnzahlKanten_Min', 'AnzahlKanten_Max',
    'MaxTiefe_Durchschnitt', 'MaxTiefe_Median', 'MaxTiefe_Std', 'MaxTiefe_Min', 'MaxTiefe_Max'
]

print(f"\n=== Statistiken für {focus_feature} ===\n")
print(annotator_stats)
print("\n")

# Gesamtstatistik
print("=== Gesamtstatistik ===")
print(f"Durchschnittliche Events pro Annotation: {df_task2_annotation['event_count'].mean():.2f}")
print(f"Median Events pro Annotation: {df_task2_annotation['event_count'].median():.2f}")
print(f"Gesamtzahl Events: {df_task2_annotation['event_count'].sum():.0f}")
print(f"Gesamtzahl Annotationen: {len(df_task2_annotation)}")
print(f"\n--- DAG-Metriken (nur Annotationen mit DAG-Strukturen) ---")
print(f"Anzahl Annotationen mit DAG: {len(df_with_chains)}")
if len(df_with_chains) > 0:
    print(f"Durchschnittliche Anzahl Kanten: {df_with_chains['num_edges'].mean():.2f}")
    print(f"Median Anzahl Kanten: {df_with_chains['num_edges'].median():.2f}")
    print(f"Durchschnittliche max. Pfadtiefe: {df_with_chains['max_depth'].mean():.2f}")
    print(f"Median max. Pfadtiefe: {df_with_chains['max_depth'].median():.2f}")
else:
    print("Keine DAG-Strukturen gefunden")

# Statistiken pro Annotator einzeln
print("\n\n=== Detaillierte Statistiken pro Annotator ===\n")
for annotator in sorted(df_task2_annotation['annotator'].unique()):
    df_ann = df_task2_annotation[df_task2_annotation['annotator'] == annotator]
    df_ann_with_chains = df_ann[df_ann['max_depth'] > 0]
    
    print(f"--- Annotator {annotator} ---")
    print(f"Anzahl Annotationen: {len(df_ann)}")
    print(f"Durchschnittliche Events: {df_ann['event_count'].mean():.2f}")
    print(f"Median Events: {df_ann['event_count'].median():.2f}")
    print(f"Min/Max Events: {df_ann['event_count'].min():.0f} / {df_ann['event_count'].max():.0f}")
    print(f"Gesamt Events: {df_ann['event_count'].sum():.0f}")
    
    print(f"\nDAG-Metriken (nur Annotationen mit DAG):")
    print(f"  Anzahl Annotationen mit DAG: {len(df_ann_with_chains)}")
    if len(df_ann_with_chains) > 0:
        print(f"  Durchschnittliche Anzahl Kanten: {df_ann_with_chains['num_edges'].mean():.2f}")
        print(f"  Median Anzahl Kanten: {df_ann_with_chains['num_edges'].median():.2f}")
        print(f"  Min/Max Kanten: {df_ann_with_chains['num_edges'].min():.0f} / {df_ann_with_chains['num_edges'].max():.0f}")
        print(f"  Durchschnittliche max. Pfadtiefe: {df_ann_with_chains['max_depth'].mean():.2f}")
        print(f"  Median max. Pfadtiefe: {df_ann_with_chains['max_depth'].median():.2f}")
        print(f"  Min/Max Pfadtiefe: {df_ann_with_chains['max_depth'].min():.0f} / {df_ann_with_chains['max_depth'].max():.0f}")
    else:
        print(f"  Keine DAG-Strukturen")
    print()


