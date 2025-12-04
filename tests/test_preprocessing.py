"""
Tests unitarios para el preprocesamiento de datos
Cubre limpieza, creación de features y encoding
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Ajustar el path para encontrar los módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cleaning import clean_data
from features import create_features


class TestCleaning:
    """Tests para la función de limpieza de datos"""
    
    @pytest.fixture
    def sample_raw_data(self):
        """Datos de prueba sin limpiar"""
        return pd.DataFrame({
            'match_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'team_color': ['blue', 'orange', 'blue'],
            'game_mode': ['Duel', 'Doubles', 'Standard'],
            'winner': ['blue', 'orange', 'blue'],
            'goal_difference': [2, -1, 3],
            'match_duration': [300, 360, 330],
            'overtime': [False, True, False]
        })
    
    def test_clean_data_converts_dates(self, sample_raw_data):
        """Verifica que las fechas se conviertan a datetime"""
        result = clean_data(sample_raw_data.copy())
        
        assert pd.api.types.is_datetime64_any_dtype(result['match_date'])
        assert result['match_date'].dtype == 'datetime64[ns]'
    
    def test_clean_data_converts_string_types(self, sample_raw_data):
        """Verifica que las columnas de texto sean strings"""
        result = clean_data(sample_raw_data.copy())
        
        assert result['team_color'].dtype == 'object'
        assert result['game_mode'].dtype == 'object'
        assert result['winner'].dtype == 'object'
    
    def test_clean_data_preserves_data_integrity(self, sample_raw_data):
        """Verifica que no se pierdan datos durante la limpieza"""
        original_shape = sample_raw_data.shape
        result = clean_data(sample_raw_data.copy())
        
        assert result.shape == original_shape
        assert len(result) == len(sample_raw_data)
    
    def test_clean_data_handles_missing_dates(self):
        """Verifica el manejo de fechas faltantes"""
        df = pd.DataFrame({
            'match_date': ['2024-01-01', None, '2024-01-03'],
            'team_color': ['blue', 'orange', 'blue'],
            'game_mode': ['Duel', 'Doubles', 'Standard'],
            'winner': ['blue', 'orange', 'blue']
        })
        
        result = clean_data(df.copy())
        
        # Verifica que se convierta a datetime pero mantenga NaT para valores nulos
        assert pd.api.types.is_datetime64_any_dtype(result['match_date'])
        assert pd.isna(result['match_date'].iloc[1])
    
    def test_clean_data_returns_dataframe(self, sample_raw_data):
        """Verifica que retorne un DataFrame"""
        result = clean_data(sample_raw_data.copy())
        
        assert isinstance(result, pd.DataFrame)


class TestFeatures:
    """Tests para la creación de features"""
    
    @pytest.fixture
    def sample_clean_data(self):
        """Datos limpios para crear features"""
        return pd.DataFrame({
            'team_color': ['Blue', 'Orange', 'Blue', 'Orange'],
            'game_mode': ['Duel', 'Doubles', 'Standard', 'Duel'],
            'goal_difference': [-5, -1, 0, 4],
            'match_duration': [280, 350, 400, 450],
            'overtime': [False, True, False, True],
            'winner': ['Orange', 'Orange', 'Blue', 'Orange']
        })
    
    def test_create_features_adds_goal_diff_category(self, sample_clean_data):
        """Verifica que se cree la categoría de diferencia de goles"""
        result = create_features(sample_clean_data.copy())
        
        assert 'goal_diff_category' in result.columns
        expected_categories = ['large_loss', 'small_loss', 'close', 'large_win']
        assert result['goal_diff_category'].dtype.name == 'category'
    
    def test_create_features_goal_diff_categories_correct(self, sample_clean_data):
        """Verifica que las categorías de diferencia de goles sean correctas"""
        result = create_features(sample_clean_data.copy())
        
        assert result['goal_diff_category'].iloc[0] == 'large_loss'  # -5
        assert result['goal_diff_category'].iloc[1] == 'small_loss'  # -1
        assert result['goal_diff_category'].iloc[2] == 'close'        # 0
        assert result['goal_diff_category'].iloc[3] == 'large_win'    # 4
    
    def test_create_features_adds_duration_bucket(self, sample_clean_data):
        """Verifica que se cree el bucket de duración"""
        result = create_features(sample_clean_data.copy())
        
        assert 'duration_bucket' in result.columns
        assert result['duration_bucket'].dtype.name == 'category'
    
    def test_create_features_duration_buckets_correct(self, sample_clean_data):
        """Verifica que los buckets de duración sean correctos"""
        result = create_features(sample_clean_data.copy())
        
        assert result['duration_bucket'].iloc[0] == 'short'      # 280
        assert result['duration_bucket'].iloc[1] == 'normal'     # 350
        assert result['duration_bucket'].iloc[2] == 'long'       # 400
        assert result['duration_bucket'].iloc[3] == 'very_long'  # 450
    
    def test_create_features_adds_is_competitive(self, sample_clean_data):
        """Verifica que se cree el indicador de partida competitiva"""
        result = create_features(sample_clean_data.copy())
        
        assert 'is_competitive' in result.columns
        assert result['is_competitive'].dtype in [np.int64, np.int32, int]
    
    def test_create_features_is_competitive_correct(self, sample_clean_data):
        """Verifica que el indicador competitivo sea correcto"""
        result = create_features(sample_clean_data.copy())
        
        assert result['is_competitive'].iloc[0] == 0  # diff=-5, no competitivo
        assert result['is_competitive'].iloc[1] == 1  # diff=-1, competitivo
        assert result['is_competitive'].iloc[2] == 1  # diff=0, competitivo
        assert result['is_competitive'].iloc[3] == 0  # diff=4, no competitivo
    
    def test_create_features_adds_team_mode(self, sample_clean_data):
        """Verifica que se cree la feature combinada team_mode"""
        result = create_features(sample_clean_data.copy())
        
        assert 'team_mode' in result.columns
        assert result['team_mode'].dtype == 'object'
    
    def test_create_features_team_mode_correct(self, sample_clean_data):
        """Verifica que team_mode tenga el formato correcto"""
        result = create_features(sample_clean_data.copy())
        
        assert result['team_mode'].iloc[0] == 'Blue_Duel'
        assert result['team_mode'].iloc[1] == 'Orange_Doubles'
        assert result['team_mode'].iloc[2] == 'Blue_Standard'
    
    def test_create_features_preserves_original_columns(self, sample_clean_data):
        """Verifica que se mantengan todas las columnas originales"""
        original_cols = set(sample_clean_data.columns)
        result = create_features(sample_clean_data.copy())
        
        assert original_cols.issubset(set(result.columns))
    
    def test_create_features_returns_dataframe(self, sample_clean_data):
        """Verifica que retorne un DataFrame"""
        result = create_features(sample_clean_data.copy())
        
        assert isinstance(result, pd.DataFrame)
    
    def test_create_features_handles_extreme_values(self):
        """Verifica el manejo de valores extremos"""
        df = pd.DataFrame({
            'team_color': ['Blue'],
            'game_mode': ['Duel'],
            'goal_difference': [50],  # Valor extremo
            'match_duration': [1000],  # Duración muy larga
            'overtime': [True],
            'winner': ['Blue']
        })
        
        result = create_features(df.copy())
        
        assert 'goal_diff_category' in result.columns
        assert 'duration_bucket' in result.columns
        # Los valores extremos deben caer en las categorías más altas
        assert result['goal_diff_category'].iloc[0] == 'large_win'
        assert result['duration_bucket'].iloc[0] == 'very_long'


class TestDataIntegration:
    """Tests de integración para el pipeline completo de preprocesamiento"""
    
    def test_full_preprocessing_pipeline(self):
        """Verifica que el pipeline completo funcione correctamente"""
        # Crear datos de prueba
        raw_data = pd.DataFrame({
            'match_date': ['2024-01-01', '2024-01-02'],
            'team_color': ['blue', 'orange'],
            'game_mode': ['Duel', 'Doubles'],
            'winner': ['blue', 'orange'],
            'goal_difference': [2, -1],
            'match_duration': [300, 360],
            'overtime': [False, True]
        })
        
        # Aplicar limpieza
        clean_df = clean_data(raw_data.copy())
        
        # Aplicar features
        final_df = create_features(clean_df)
        
        # Verificaciones finales
        assert len(final_df) == 2
        assert 'goal_diff_category' in final_df.columns
        assert 'duration_bucket' in final_df.columns
        assert 'is_competitive' in final_df.columns
        assert 'team_mode' in final_df.columns
        assert pd.api.types.is_datetime64_any_dtype(final_df['match_date'])
    
    def test_preprocessing_maintains_data_quality(self):
        """Verifica que el preprocesamiento mantenga la calidad de los datos"""
        raw_data = pd.DataFrame({
            'match_date': ['2024-01-01'] * 10,
            'team_color': ['blue'] * 5 + ['orange'] * 5,
            'game_mode': ['Duel'] * 10,
            'winner': ['blue'] * 5 + ['orange'] * 5,
            'goal_difference': list(range(-4, 6)),
            'match_duration': list(range(280, 480, 20)),
            'overtime': [False] * 10
        })
        
        clean_df = clean_data(raw_data.copy())
        final_df = create_features(clean_df)
        
        # No debe haber valores nulos en las nuevas features
        assert final_df['goal_diff_category'].notna().all()
        assert final_df['duration_bucket'].notna().all()
        assert final_df['is_competitive'].notna().all()
        assert final_df['team_mode'].notna().all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])