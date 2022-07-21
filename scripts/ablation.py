import sys

fp = sys.argv[1]
with open(fp) as file:
    lines = [
        line.strip() for line in file.read().strip().splitlines()
        if line.strip()]

datasets = ['pubmed', 'flickr', 'ppi', 'arxiv', 'yelp', 'reddit']
labels = ['Pubmed', 'Flickr', 'PPI', 'ogbn-arxiv', 'Yelp', 'Reddit']
bests = [{}, {}]
log = dataset = uniform = skip = embedding = attention = None
scores = []
for line in lines + [' Namespace(']:
    if ' Namespace(' in line:
        if scores:
            key = (dataset, uniform, skip, embedding, attention)
            if not (dataset == 'reddit' and log['precompute'] == '0'):
                for i in range(2):
                    if float(scores[i].split('±', 1)[0]) > float(
                            bests[i].get(key, ['', '0'])[1].split('±', 1)[0]):
                        bests[i][key] = [log, scores[i]]
            scores = []
        if 'ECN' not in line:
            continue
        dataset = line.split("dataset='", 1)[-1].split("'", 1)[0]
        uniform = 'no_importance_sampling=True' in line
        skip = 'skip_connection=True' in line
        embedding = 'embedding=0.0,' not in line
        attention = 'attention=0,' not in line
        log = dict(
            kvs.strip().split('=')
            for kvs in line.split('Namespace(', 1)[1][:-1].split(','))
        log['dataset'] = log['dataset'].strip("'")
        log['importance_sampling'] = str(not uniform)
    elif line.startswith('score:'):
        scores.append(line.split(':', 1)[-1].strip())
cols = (
    'dataset importance_sampling skip_connection '
    'embedding attention precompute').split()
rows = sorted([
    ' & '.join(list(map(log.get, cols)) + [name, score]) + r' \\'
    for name, scores in zip(['GCN', 'ECN'], bests)
    for log, score in scores.values()])
with open(
        'output/%s.tex' % (fp.split('/')[-1].rsplit('.', 1)[0]), 'w') as file:
    file.write('Dataset & Importance Sampling & Skip Connection & Embedding '
               '& Attention & Precompute & Hidden Dim & Middle Layers '
               '& Dropout & Weight Decay & Evaluation Model & Score \\\\\n')
    file.write('\n'.join(rows))

effects = {dataset: [[[], []] for _ in range(5)] for dataset in datasets}
pres = {dataset: [[], []] for dataset in datasets}
with open('output/ablation_detail.tex', 'w') as file:
    file.write(
        ' & Importance & Skip & Embedding & Attention & Unbiased & '
        + ' & '.join(labels) + r' \\' + '\n')
    file.write(
        ' & Sampling & Connection & & & Inference & '
        + ' & '.join('' for _ in labels) + r' \\' + '\n')
    file.write(r'\hline' + '\n')
    for uniform in (True, False):
        for skip in (False, True):
            for embedding in (False, True):
                for attention in (False, True):
                    for i in range(2):
                        flags = (
                            not uniform, skip, embedding, attention, bool(i))
                        row = [r'\checkmark' if flag else '' for flag in flags]
                        for dataset in datasets:
                            key = (
                                dataset, uniform, skip, embedding, attention)
                            if key not in bests[i]:
                                row.append('|')
                                continue
                            log, score = bests[i][key]
                            val = score.split('±')[0]
                            row.append('%s (%s)' % (val, log['precompute']))
                            for flag, effect in zip(flags, effects[dataset]):
                                effect[int(flag)].append(float(val))
                            pres[dataset][int(skip)].append(
                                int(log['precompute']))
                        file.write(
                            r'\rno & ' + ' & '.join(row) + r' \\' + '\n')

with open('output/ablation_attention.tex', 'w') as file:
    file.write(
        'Importance & Embedding &' + ' & '.join(labels[:-1]) + r' \\' + '\n')
    file.write(
        'Sampling & & ' + ' & '.join('' for _ in labels[:-1]) + r' \\' + '\n')
    file.write(r'\hline' + '\n')
    vals = {}
    alts = {}
    for uniform in (True, False):
        for embedding in (False, True):
            row = [
                r'\checkmark' if flag else ''
                for flag in (not uniform, embedding)]
            for dataset in datasets:
                key = (dataset, uniform, False, embedding, True)
                val = float(bests[1][key][1].split('±')[0])
                alt = [
                    float(bests[i][(
                        dataset, uniform, False, embedding, attention
                    )][1].split('±')[0])
                    if (attention, i) != (True, 1) else 0
                    for attention in (False, True)
                    for i in range(2)
                ]
                vals.setdefault(dataset, []).append(val)
                alts.setdefault(dataset, []).extend(alt)
                # alt = sum(alt) / len(alt)
                alt = max(alt, default=0)
                row.append('%+.2f' % (val - alt))
            file.write(' & '.join(row) + r' \\' + '\n')
    file.write(' & '.join(
        '%+.2f' % (
            max(vals[dataset]) - max(alts[dataset])
            # sum(vals[dataset]) / len(vals[dataset])
            # - sum(alts[dataset]) / len(alts[dataset])
        ) for dataset in datasets))

with open('output/ablation_summary.tex', 'w') as file:
    file.write(
        ' & Importance & Skip & Embedding & Attention & Unbiased '
        + r'\\' + '\n')
    file.write(
        ' & Sampling & Connection & & & Inference '
        + r'\\' + '\n')
    file.write(r'\hline' + '\n')
    for dataset, label in zip(datasets, labels):
        file.write('%s & %s \\\\\n' % (label, ' & '.join(
            (
                '%+.2f' % (
                    # float(sum(ts) - sum(fs)) / len(ts)
                    max(ts, default=0) - max(fs, default=0)
                ) + (' (%+.2f)' % (
                    float(sum(pres[dataset][1]) - sum(pres[dataset][0]))
                    / len(pres[dataset][0])
                ) if i == 1 else '')
            )
            if len(fs) == len(ts) > 0 else '|'
            for i, (fs, ts) in enumerate(effects[dataset]))))
