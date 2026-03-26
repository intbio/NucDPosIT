#!/usr/bin/env python3
import sys, os
import pandas as pd

def convert_to_bed(input_file=None, output_file=None):
    """
    Конвертирует таблицу с мотивами в формат BED.
    
    Формат входных данных (пример заголовка):
    motif_id motif_alt_id sequence_name start stop strand score p-value q-value matched_sequence
    
    Формат BED:
    chrom (sequence_name) start end name (motif_alt_id) score strand
    """
    
    # Если файл не указан, используем стандартный ввод
    if input_file:
        input_file = os.path.abspath(input_file)
        df = pd.read_csv(input_file, sep='\t')
    else:
        df = pd.read_csv(sys.stdin, sep='\t')
    df = df.dropna()
    
    # Создаем DataFrame в формате BED
    bed_df = pd.DataFrame({
        'chrom': df['sequence_name'],
        'start': df['start'].astype(int),
        'end': df['stop'].astype(int),
        'name': df['motif_alt_id'],
        'score': df['score'],
        'strand': df['strand']
    })
    
    # Если выходной файл указан, записываем в него
    if output_file:
        output_file = os.path.abspath(output_file)
        bed_df.to_csv(output_file, sep='\t', index=False, header=False)
        print(f"Файл сохранен как: {output_file}")
    else:
        # Иначе выводим в stdout
        bed_df.to_csv(sys.stdout, sep='\t', index=False, header=False)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Конвертировать таблицу с мотивами в формат BED')
    parser.add_argument('-i', '--input', help='Входной файл (по умолчанию: stdin)')
    parser.add_argument('-o', '--output', help='Выходной файл (по умолчанию: stdout)')
    
    args = parser.parse_args()
    
    convert_to_bed(args.input, args.output)