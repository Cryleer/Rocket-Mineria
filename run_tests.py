#!/usr/bin/env python
"""
Script para ejecutar tests del proyecto Rocket Mineria
Proporciona diferentes opciones de ejecución
"""

import subprocess
import sys
import argparse


def run_command(cmd):
    """Ejecuta un comando y muestra la salida"""
    print(f"\n{'='*60}")
    print(f"Ejecutando: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def run_all_tests():
    """Ejecuta todos los tests"""
    return run_command(['pytest', 'tests/', '-v'])


def run_preprocessing_tests():
    """Ejecuta solo tests de preprocesamiento"""
    return run_command(['pytest', 'tests/test_preprocessing.py', '-v'])


def run_model_tests():
    """Ejecuta solo tests del modelo"""
    return run_command(['pytest', 'tests/test_model.py', '-v'])


def run_api_tests():
    """Ejecuta solo tests de la API"""
    return run_command(['pytest', 'tests/test_api.py', '-v'])


def run_with_coverage():
    """Ejecuta tests con reporte de cobertura"""
    return run_command([
        'pytest',
        'tests/',
        '--cov=src',
        '--cov=api',
        '--cov-report=term-missing',
        '--cov-report=html',
        '-v'
    ])


def run_quick_tests():
    """Ejecuta tests rápidos (sin los marcados como slow)"""
    return run_command(['pytest', 'tests/', '-v', '-m', 'not slow'])


def run_specific_test(test_path):
    """Ejecuta un test específico"""
    return run_command(['pytest', test_path, '-v', '-s'])


def main():
    parser = argparse.ArgumentParser(
        description='Ejecutar tests del proyecto Rocket Mineria'
    )
    parser.add_argument(
        'test_type',
        nargs='?',
        choices=['all', 'preprocessing', 'model', 'api', 'coverage', 'quick', 'specific'],
        default='all',
        help='Tipo de tests a ejecutar'
    )
    parser.add_argument(
        '--path',
        type=str,
        help='Ruta específica del test (solo con test_type=specific)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("TESTS UNITARIOS - PROYECTO ROCKET MINERIA")
    print("="*60)
    
    if args.test_type == 'all':
        print("\n📋 Ejecutando todos los tests...")
        exit_code = run_all_tests()
    
    elif args.test_type == 'preprocessing':
        print("\n🧹 Ejecutando tests de preprocesamiento...")
        exit_code = run_preprocessing_tests()
    
    elif args.test_type == 'model':
        print("\n🤖 Ejecutando tests del modelo...")
        exit_code = run_model_tests()
    
    elif args.test_type == 'api':
        print("\n🌐 Ejecutando tests de la API...")
        exit_code = run_api_tests()
    
    elif args.test_type == 'coverage':
        print("\n📊 Ejecutando tests con cobertura...")
        exit_code = run_with_coverage()
        print("\n📄 Reporte HTML generado en: htmlcov/index.html")
    
    elif args.test_type == 'quick':
        print("\n⚡ Ejecutando tests rápidos...")
        exit_code = run_quick_tests()
    
    elif args.test_type == 'specific':
        if not args.path:
            print("\n❌ Error: Debes especificar --path para ejecutar un test específico")
            sys.exit(1)
        print(f"\n🎯 Ejecutando test específico: {args.path}")
        exit_code = run_specific_test(args.path)
    
    print("\n" + "="*60)
    if exit_code == 0:
        print("✅ TESTS COMPLETADOS EXITOSAMENTE")
    else:
        print("❌ ALGUNOS TESTS FALLARON")
    print("="*60 + "\n")
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
