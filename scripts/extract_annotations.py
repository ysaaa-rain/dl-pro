"""Read only the relevant tables; preserve workbook and source coordinates."""
import argparse
import json
from pathlib import Path
import openpyxl
import pandas as pd
from dlpro.data import validate_annotations


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workbook', default='data/annotations/标注数据.xlsx')
    p.add_argument('--output', default='data/annotations/structures_source.csv')
    a = p.parse_args()
    w = openpyxl.load_workbook(a.workbook, read_only=True, data_only=True)
    rows = []
    for number, label in [(4, 'OPCID'), (5, 'CHIN'), (6, 'CHID')]:
        sheet = w[f'Supplementary Table {number}']
        values = sheet.values
        header = next(values)
        for row_number, row in enumerate(values, 2):
            if row[0] is None:
                continue
            item = dict(zip(header, row))
            if item['Chr'] != 'MG1655':
                raise ValueError('Unexpected reference alias')
            rows.append(dict(structure_id=row[0], type=label, chrom='NC_000913.3',
                start=item['Start'], end=item['End'], source_chrom=item['Chr'],
                source_center=item.get('Center'), source_sheet=sheet.title, source_row=row_number,
                coordinate_status='source_preserved_origin_convention_unconfirmed'))
    frame = validate_annotations(pd.DataFrame(rows), {'NC_000913.3': 4641652})
    # No silent +/-1 conversion: provisional table, not frozen training annotations.
    frame.to_csv(a.output, index=False)
    stats = dict(total=len(frame), counts=frame.type.value_counts().to_dict(),
        coordinate_status='Coordinates copied exactly; base convention not established by headers',
        reference_mapping='MG1655 -> NC_000913.3; course reference and both Cooler axes agree',
        source=a.workbook, output=a.output)
    Path('results/annotation_audit.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
