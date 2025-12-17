"""
Module de sélection avancée de modèles.

Fournit différentes stratégies pour sélectionner le meilleur modèle
en fonction de critères variés (performance, vitesse, compromis).
"""
from typing import List, Dict
import numpy as np


class ModelSelector:
    """
    Sélection de modèle avec différentes stratégies.
    """

    @staticmethod
    def select_by_score(results: List[Dict], metric: str = 'valid_score'):
        """
        Sélection par meilleur score.

        Args:
            results: liste des résultats des modèles
            metric: métrique à utiliser ('valid_score' ou 'train_score')

        Returns:
            str: nom du meilleur modèle

        Raises:
            ValueError: si aucun résultat valide
        """
        if not results:
            raise ValueError("Aucun résultat à analyser")

        # Filtrer les résultats avec la métrique disponible
        valid_results = [r for r in results if r.get(metric) is not None]

        if not valid_results:
            raise ValueError(f"Aucun modèle n'a de score valide pour '{metric}'")

        sorted_results = sorted(valid_results, key=lambda x: x[metric], reverse=True)
        return sorted_results[0]['name']

    @staticmethod
    def select_by_speed_score_tradeoff(results: List[Dict],
                                       score_weight: float = 0.7,
                                       speed_weight: float = 0.3,
                                       metric: str = 'valid_score'):
        """
        Sélection avec compromis vitesse/performance.

        Cette méthode normalise les scores et les temps d'entraînement,
        puis calcule un score combiné pondéré.

        Args:
            results: résultats des modèles
            score_weight: poids du score (0-1)
            speed_weight: poids de la vitesse (0-1)
            metric: métrique de score à utiliser

        Returns:
            str: nom du meilleur modèle selon le compromis

        Raises:
            ValueError: si les poids ne sont pas valides ou si pas de résultats
        """
        if not results:
            raise ValueError("Aucun résultat à analyser")

        if not (0 <= score_weight <= 1 and 0 <= speed_weight <= 1):
            raise ValueError("Les poids doivent être entre 0 et 1")

        if abs(score_weight + speed_weight - 1.0) > 1e-6:
            raise ValueError("La somme des poids doit être égale à 1")

        # Filtrer les résultats valides
        valid_results = [r for r in results
                        if r.get(metric) is not None and r.get('training_time') is not None]

        if not valid_results:
            raise ValueError(f"Pas assez de résultats valides avec '{metric}' et 'training_time'")

        # Extraire les scores et temps
        scores = np.array([r[metric] for r in valid_results])
        times = np.array([r['training_time'] for r in valid_results])

        # Normalisation min-max
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        # Pour le temps, on inverse : temps court = bon
        times_norm = 1 - (times - times.min()) / (times.max() - times.min() + 1e-8)

        # Score combiné
        combined_scores = score_weight * scores_norm + speed_weight * times_norm

        best_idx = np.argmax(combined_scores)
        return valid_results[best_idx]['name']

    @staticmethod
    def select_top_k(results: List[Dict], k: int = 3, metric: str = 'valid_score'):
        """
        Retourne les k meilleurs modèles (pour ensemble ultérieur).

        Args:
            results: résultats des modèles
            k: nombre de modèles à retourner
            metric: métrique à utiliser pour le tri

        Returns:
            List[str]: noms des k meilleurs modèles

        Raises:
            ValueError: si k est invalide ou pas assez de résultats
        """
        if not results:
            raise ValueError("Aucun résultat à analyser")

        if k <= 0:
            raise ValueError("k doit être positif")

        # Filtrer les résultats valides
        valid_results = [r for r in results if r.get(metric) is not None]

        if not valid_results:
            raise ValueError(f"Aucun modèle n'a de score valide pour '{metric}'")

        # Limiter k au nombre de résultats disponibles
        k = min(k, len(valid_results))

        sorted_results = sorted(valid_results, key=lambda x: x[metric], reverse=True)
        return [r['name'] for r in sorted_results[:k]]

    @staticmethod
    def select_by_overfitting_control(results: List[Dict],
                                      max_gap: float = 0.1):
        """
        Sélectionne le meilleur modèle en évitant le surapprentissage.

        Sélectionne le modèle avec le meilleur score de validation parmi
        ceux dont l'écart train-validation est inférieur à max_gap.

        Args:
            results: résultats des modèles
            max_gap: écart maximum acceptable entre train et validation

        Returns:
            str: nom du modèle sélectionné

        Raises:
            ValueError: si pas de résultats valides
        """
        if not results:
            raise ValueError("Aucun résultat à analyser")

        # Filtrer les modèles avec train et valid scores
        valid_results = [r for r in results
                        if r.get('train_score') is not None
                        and r.get('valid_score') is not None]

        if not valid_results:
            raise ValueError("Aucun modèle n'a à la fois train_score et valid_score")

        # Filtrer par gap
        filtered_results = [
            r for r in valid_results
            if abs(r['train_score'] - r['valid_score']) <= max_gap
        ]

        # Si aucun modèle ne passe le filtre, prendre celui avec le plus petit gap
        if not filtered_results:
            filtered_results = sorted(valid_results,
                                     key=lambda x: abs(x['train_score'] - x['valid_score']))[:1]

        # Sélectionner le meilleur valid_score parmi les modèles filtrés
        best = max(filtered_results, key=lambda x: x['valid_score'])
        return best['name']

    @staticmethod
    def get_model_rankings(results: List[Dict], metric: str = 'valid_score'):
        """
        Retourne le classement complet des modèles.

        Args:
            results: résultats des modèles
            metric: métrique pour le classement

        Returns:
            List[Dict]: résultats triés avec rang ajouté
        """
        if not results:
            return []

        # Filtrer et trier
        valid_results = [r for r in results if r.get(metric) is not None]
        sorted_results = sorted(valid_results, key=lambda x: x[metric], reverse=True)

        # Ajouter le rang
        for i, result in enumerate(sorted_results, 1):
            result['rank'] = i

        return sorted_results
