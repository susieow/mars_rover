"""Search recorded experiments before proposing an identical run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def matches(row, args):
    checks = ((args.friction, row.get('friction')), (args.trigger, row.get('trigger')),
              (args.speed, row.get('speed')), (args.direction, row.get('direction')),
              (args.scope, row.get('scope')))
    return (all(want is None or want == actual for want, actual in checks)
            and (not args.crossed or row.get('local_crossing') is True)
            and (not args.id or args.id in row['id'])
            and (not args.text or args.text.lower() in json.dumps(row, ensure_ascii=False).lower()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--friction', type=int, choices=(4, 6, 8))
    parser.add_argument('--trigger', choices=('contact', 'stall'))
    parser.add_argument('--speed', type=int, choices=tuple(range(3, 16)))
    parser.add_argument('--direction', choices=('SN', 'SP'))
    parser.add_argument('--scope', choices=('final', 'history'))
    parser.add_argument('--id')
    parser.add_argument('--text')
    parser.add_argument('--crossed', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    rows = [json.loads(line) for line in (ROOT / 'docs/experiment_index.jsonl').read_text(encoding='utf-8').splitlines() if line]
    selected = [r for r in rows if matches(r, args)]
    if args.json:
        print(json.dumps(selected, ensure_ascii=False, indent=2))
        return
    print(f'匹配 {len(selected)} 条已记录经验。无匹配仅表示本索引未查到，不代表可自动开跑。')
    for row in selected:
        outcome = '曾局部越阶' if row.get('local_crossing') else '未记录局部越阶'
        print(f"\n{row['key']} [{row['scope']}] {outcome}")
        if row.get('scope') == 'final':
            print(f"  摩擦 {row['friction']} / {row['trigger']} / {row['speed']} rad/s / {row['direction']}")
        print('  记录原因：' + ('、'.join(row.get('process_reasons', [])) or row.get('termination_reason') or '见原始结果'))
        print('  证据：' + (row.get('source_result') or row.get('source_manifest') or '见索引'))
        print('  视频：' + (row.get('video_path') or '见原始媒体清单'))
    if not selected and args.friction is not None and args.trigger and args.speed is not None:
        print('当前参数属于最终版本范围；请核对 EXPERIENCE.md 的已测/未测表及原始数据。')


if __name__ == '__main__':
    main()
