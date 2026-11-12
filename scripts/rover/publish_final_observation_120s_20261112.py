"""Publish verified 120-second extensions while retaining original evidence."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/final_observation_120s_20261112'
BATCH = RUN.name


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def check_markdown(path):
    data = path.read_bytes()
    text = data.decode('utf-8')
    assert not data.startswith(b'\xef\xbb\xbf')
    assert '\ufffd' not in text and '?????' not in text
    assert any('\u4e00' <= c <= '\u9fff' for c in text)


def main():
    manifest = json.loads((RUN / 'manifest.json').read_text())
    progress = json.loads((RUN / 'progress.json').read_text())
    assert progress['complete'] and not progress['failed']
    assert len(progress['simulation_completed']) == len(progress['media_completed']) == 17
    catalog_path = ROOT / 'docs/final_experiments.json'
    catalog = json.loads(catalog_path.read_text())
    assert digest(RUN / 'parent_catalog.json') == manifest['parent_catalog_sha256']
    originals = {r['id']: r for r in manifest['cases']}
    assert len(originals) == 17
    index_path = ROOT / 'docs/experiment_index.jsonl'
    index = [json.loads(line) for line in index_path.read_text().splitlines() if line]
    by_key = {r['key']: r for r in index}
    replacements, summaries = {}, []
    additions = []
    for identifier, parent in originals.items():
        folder = RUN / 'cases' / identifier
        result_path = folder / 'result.json'
        result = json.loads(result_path.read_text())
        verification = json.loads((folder / 'verification.json').read_text())
        prefix = json.loads((folder / 'prefix_verification.json').read_text())
        media = json.loads((folder / 'media.json').read_text())
        media_check = json.loads((folder / 'media_verification.json').read_text())
        assert verification['passed'] and prefix['passed'] and media_check['passed']
        assert abs(result['actual_duration_s'] - 120) < 1e-7 and media['frames'] == 2401
        assert digest(result_path) == verification['result_sha256']
        assert digest(folder / 'trajectory.npz') == result['trajectory_sha256'] == media['trajectory_sha256']
        assert digest(media['video_path']) == media['video_sha256']
        assert digest(parent['result_path']) == parent['result_sha256']
        row = deepcopy(parent)
        row.update(key=BATCH + '/' + identifier, batch=BATCH, parameters=result['parameters'],
                   first_contact_s=result['first_contact_s'], trigger_s=result['stall_trigger_s'],
                   end_s=result['actual_duration_s'], first_failure_s=result['first_failure_s'],
                   crossing=result['true_both_rear_crossing'], first_crossing_s=result['first_rear_crossing_s'],
                   highest_level=result['rear_highest_100ms_geometry_level'],
                   retained_level=result['rear_retained_geometry_level_final_1s'],
                   platform=result['ascent_platform_support'], reasons=result['process_reasons'],
                   penetration_mm=result['audit']['max_contact_penetration_m'] * 1000,
                   overspeed=result['overspeed'], termination_reason=result['termination_reason'],
                   result_path=str(result_path), result_sha256=digest(result_path),
                   trajectory_path=str(folder / 'trajectory.npz'), trajectory_sha256=result['trajectory_sha256'],
                   curves_path=str(folder / 'curves.json'), video_path=media['video_path'],
                   video_sha256=media['video_sha256'], full_route_evaluated=result['full_route_evaluated'],
                   dataset_accepted=result['dataset_accepted'], extended_to_120s=True,
                   parent_key=parent['key'], parent_end_s=parent['end_s'],
                   parent_highest_level=parent['highest_level'])
        assert row['highest_level'] >= parent['highest_level'] and row['crossing']
        replacements[identifier] = row
        old_index = by_key[parent['key']]
        new_index = deepcopy(old_index)
        new_index.pop('superseded_by', None)
        new_index.pop('replacement_reason', None)
        new_index.update(key=row['key'], batch=BATCH, scope='final', parameters=row['parameters'],
                         source_result=str(result_path), source_result_sha256=row['result_sha256'],
                         trajectory_sha256=row['trajectory_sha256'], scene_sha256=result['scene_sha256'],
                         source_sha256=result['source_sha256'], process_reasons=result['process_reasons'],
                         process_valid=result['process_valid'], termination_reason=result['termination_reason'],
                         local_crossing=row['crossing'], full_route_evaluated=result['full_route_evaluated'],
                         full_route_success=result['full_route_success'], dataset_accepted=result['dataset_accepted'],
                         video_path=row['video_path'], video_sha256=row['video_sha256'],
                         registries=[{'file': 'docs/final_observation_120s_20261112.json', 'case': identifier}],
                         observation_extension=result['observation_extension'], highest_level=row['highest_level'],
                         retained_level=row['retained_level'], end_s=row['end_s'])
        old_index.update(scope='history', superseded_by=row['key'],
                         replacement_reason='Same parameters and identical prefix; observation extended to absolute simulation time 120 s.')
        if row['key'] not in by_key:
            additions.append(new_index)
        else:
            by_key[row['key']].update(new_index)
        summaries.append({**verification, 'friction': row['friction'], 'trigger': row['trigger'],
                          'speed': row['speed'], 'direction': row['direction'],
                          'parent_key': parent['key'], 'key': row['key'],
                          'penetration_mm': row['penetration_mm'], 'process_reasons': row['reasons'],
                          'verification_path': str(folder / 'verification.json'),
                          'prefix_verification_path': str(folder / 'prefix_verification.json'),
                          'media_verification_path': str(folder / 'media_verification.json'),
                          'video_path': row['video_path'], 'video_sha256': row['video_sha256']})
    catalog['cases'] = [replacements.get(r['id'], r) for r in catalog['cases']]
    assert len(catalog['cases']) == 112 and sum(r['crossing'] for r in catalog['cases']) == 68
    index.extend(additions)
    technical_key = BATCH + '/technical_attempt_01'
    if technical_key not in by_key:
        index.append({'key': technical_key, 'id': 'technical_attempt_01', 'batch': BATCH, 'scope': 'history',
                      'kind': 'technical_prefix_verifier_failure', 'parameters': {'observation_end_s': 120},
                      'source_result': str(RUN / 'technical_attempt_01/failure.json'),
                      'source_result_sha256': digest(RUN / 'technical_attempt_01/failure.json'),
                      'process_reasons': ['ValueError: Out of range float values are not JSON compliant: nan'],
                      'termination_reason': 'technical_verifier_error_before_120s',
                      'local_crossing': False, 'full_route_evaluated': False, 'full_route_success': False,
                      'dataset_accepted': False, 'completed_120s_experiments': 0,
                      'note': '12 started runs; matching NaN in missing slip diagnostics was mishandled. This is one technical evidence entry, not 12 completed physical experiments.'})
    final = [r for r in index if r['scope'] == 'final']
    assert len(final) == len({r['id'] for r in final}) == 112
    assert len(index) == len({r['key'] for r in index})
    summary = {'batch': BATCH, 'completed': 17, 'absolute_end_s': 120.0,
               'improved_highest_count': sum(r['highest_level'] > r['parent_highest_level'] for r in summaries),
               'platform_count': sum(r['platform'] for r in summaries),
               'highest_level_counts': dict(sorted(Counter(r['highest_level'] for r in summaries).items())),
               'retained_level_counts': dict(sorted(Counter(r['retained_level'] for r in summaries).items())),
               'parent_catalog_sha256': manifest['parent_catalog_sha256'],
               'all_prefixes_bitwise_equal': True, 'rocker_cycles': 3, 'rocker_period_s': 8,
               'cases': summaries, 'technical_attempt': str(RUN / 'technical_attempt_01/failure.json')}
    catalog['observation_extension'] = {k:v for k,v in summary.items() if k not in ('cases', 'technical_attempt')}
    catalog['updated'] = '2026-11-12'
    write_json(ROOT / 'docs/final_observation_120s_20261112.json', summary)
    staged = index_path.with_name(index_path.name + '.tmp')
    staged.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in index), encoding='utf-8')
    staged.replace(index_path)
    write_json(catalog_path, catalog)

    readme_path = ROOT / 'README.md'
    readme = readme_path.read_text(encoding='utf-8')
    readme = readme.replace('68 条曾满足双后轮局部越阶条件，尚无登顶或完整路线成功样本',
                            '68 条曾满足双后轮局部越阶条件，完整路线合格成功样本仍为 0')
    readme = readme.replace('| 观察时长 | 触发后 30 秒；未触发最多 120 秒 |',
                            '| 观察时长 | 原最高 2/3 级的 17 条已延长至仿真总时刻 120 秒；其余保留触发后 30 秒 |')
    readme = readme.replace('浏览器页面只展示最终版本 112 条视频和曲线，可按全部参数筛选',
                            '浏览器页面展示最终版本 112 条视频和曲线，其中 17 条已替换为完整 120 秒版本；可按全部参数和最高几何等级筛选')
    marker = '\n## 延长至 120 秒的观察（2026-11-12）\n'
    if marker in readme:
        readme = readme.split(marker)[0]
    readme += marker + f'\n原最高 3 级 2 条、2 级 15 条，保持原控制和物理参数延长观察；全部通过原终点前逐步轨迹一致性核对。{summary["improved_highest_count"]} 条提高最高几何等级，{summary["platform_count"]} 条记录六轮平台支撑。详细逐条结果见 `docs/final_observation_120s_20261112.json` 和经验文档。原短版轨迹及视频保留为历史，前端相同 ID 显示延长版。\n'
    readme_path.write_text(readme, encoding='utf-8')

    experience_path = ROOT / 'EXPERIENCE.md'
    experience = experience_path.read_text(encoding='utf-8')
    experience = experience.replace('之后仅新增 1 条 5000 Hz 物理步长对照，见下文。',
                                    '之后新增 1 条 5000 Hz 对照，以及原最高 2/3 级的 17 条 120 秒延长观察，见下文。')
    experience = experience.replace('触发后观察 30 s，未触发最多 120 s。',
                                    '17 条原最高 2/3 级已延长至总时刻 120 s；其余触发后观察 30 s，未触发最多 120 s。')
    experience = experience.replace('所有 112 条均未登顶，保留过程原因。',
                                    f'当前 112 条中 {summary["platform_count"]} 条记录六轮平台支撑，保留全部过程原因；登顶和完整路线结论须分开判断。')
    experience = experience.replace('当前收录 1199 条带来源的记录', f'当前收录 {len(index)} 条带来源的记录')
    start = experience.index('| 摩擦 | 触发 | 已测轮速 rad/s |')
    end = experience.index('\n每档包含', start)
    table = ['| 摩擦 | 触发 | 已测轮速 rad/s | 条数 | 曾局部越阶 | 末 1 秒仍保持越阶等级 | 最大穿透 mm |',
             '|---:|---|---|---:|---:|---:|---:|']
    for mu in (4,6,8):
        for trigger in ('contact','stall'):
            group = [r for r in catalog['cases'] if r['friction'] == mu and r['trigger'] == trigger]
            speeds = [r['speed'] for r in group]
            table.append(f'| {mu} | {"首次接触" if trigger == "contact" else "卡住后"} | {str(min(speeds))+"～"+str(max(speeds)) if speeds else "—"} | {len(group)} | {sum(r["crossing"] for r in group) if group else "—"} | {sum(r["retained_level"]>0 for r in group) if group else "—"} | {max(r["penetration_mm"] for r in group):.2f} |' if group else f'| {mu} | 卡住后 | — | 0 | — | — | — |')
    experience = experience[:start] + '\n'.join(table) + '\n' + experience[end:]
    if marker in experience:
        experience = experience.split(marker)[0]
    for row in catalog['cases']:
        outcome = f'曾越阶（最高{row["highest_level"]}，末段{row["retained_level"]}）' if row['crossing'] else '未越阶'
        table_row = f'| `{row["id"]}` | {row["friction"]} | {"接触" if row["trigger"] == "contact" else "卡住"} | {row["speed"]} | {row["direction"]} | {outcome} | {row["penetration_mm"]:.2f} |'
        experience, count = re.subn(r'^\| `' + re.escape(row['id']) + r'` \|.*$', table_row, experience, flags=re.M)
        assert count == 1, row['id']
    experience += marker + '\n保持原初态、摩擦、轮速、触发、±20°顺序及 1 ms 步长，只取消原触发后 30 秒的结束条件，跑到总时刻 120 秒。摇臂仍为 8 秒/周期、3 周期，之后回中立；力矩限制、速度审核均保留，违规仅记录。17 条旧终点之前的状态、控制和接触数组逐步一致。\n\n'
    experience += f'最高等级提高 {summary["improved_highest_count"]}/17；六轮平台支撑 {summary["platform_count"]}/17。最高等级分布 {summary["highest_level_counts"]}，末 1 秒保持分布 {summary["retained_level_counts"]}。各条观察时间不同，当前覆盖统计使用前端最新版本；与旧表对比须留意观察窗。\n\n'
    experience += '| 记录 ID | 原最高 | 120 秒最高 | 末 1 秒 | 平台支撑 | 延长段车身净前移 m | 最后 30 秒净前移 m |\n|---|---:|---:|---:|---|---:|---:|\n'
    for r in summaries:
        experience += f'| `{r["id"]}` | {r["parent_highest_level"]} | {r["highest_level"]} | {r["retained_level"]} | {"有" if r["platform"] else "无"} | {r["additional_body_displacement"]["dx_m"]:.4f} | {r["tail_body_displacement"]["30s"]["dx_m"]:.4f} |\n'
    experience += '\n前端保持原 ID，用延长版代替短版；原结果和视频仍在原路径，经验索引标为 history 并指向新 key。逐条核验与媒体哈希见 `docs/final_observation_120s_20261112.json`。首次技术尝试在缺失滑移诊断 NaN 的比较/JSON 输出处失败，12 条已启动、0 条完成 120 秒；修正核对器后重跑成功，不把该技术失败当成车辆失败。\n'
    experience_path.write_text(experience, encoding='utf-8')
    check_markdown(readme_path)
    check_markdown(experience_path)
    print(json.dumps({k:v for k,v in summary.items() if k != 'cases'}, ensure_ascii=False))
    print(json.dumps({'indexed_records':len(index), 'markdown_encoding_verified': True}))


if __name__ == '__main__':
    main()
