"""Build the detailed PDF and Markdown report from saved, computed evidence."""
from pathlib import Path
import csv
import hashlib
import json
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results/classification'
OUT = ROOT / 'reports'
OUT.mkdir(parents=True, exist_ok=True)
info = json.loads((R / 'metadata.json').read_text())
def records(name):
    with (R / name).open() as f: return list(csv.DictReader(f))
cv = {(r['condition'], r['model']): r for r in records('cv_summary.csv')}
test = {(r['condition'], r['model']): r for r in records('test_metrics.csv')}
paired = records('paired_comparison.csv')
per = records('per_class_test.csv')
primary = info['primary_classifier']
conditions = ['local_current', 'temporal', 'spatial_current', 'spatiotemporal']
models = ['Logistic', 'LDA', 'RandomForest']
names = {'local_current': 'Local current', 'temporal': 'Temporal', 'spatial_current': 'Spatial current', 'spatiotemporal': 'Spatiotemporal', 'persistence': 'Persistence'}
styles = getSampleStyleSheet()
styles['BodyText'].fontSize = 10.2; styles['BodyText'].leading = 14.5; styles['BodyText'].spaceAfter = 9
styles['Heading1'].fontSize = 18; styles['Heading1'].leading = 23; styles['Heading1'].spaceAfter = 13
styles['Heading2'].fontSize = 12; styles['Heading2'].spaceBefore = 10; styles['Heading2'].spaceAfter = 7
styles.add(ParagraphStyle(name='Cover', fontName='Helvetica-Bold', fontSize=28, leading=34, spaceAfter=20))
styles.add(ParagraphStyle(name='Small', fontName='Helvetica', fontSize=8, leading=10.5, spaceAfter=5))
story, md = [], []
def p(value, style='BodyText'):
    story.append(Paragraph(value, styles[style])); md.append(value.replace('<b>', '').replace('</b>', '').replace('<br/>', '\n'))
def h(value): p(value, 'Heading1')
def sub(value): p(value, 'Heading2')
def page(): story.append(PageBreak()); md.append('\n---\n')
def tbl(values, widths=None):
    md.append('\n'.join(['| ' + ' | '.join(str(c).replace('<br/>', ' ') for c in row) + ' |' for row in values[:1]] + ['| ' + ' | '.join('---' for _ in values[0]) + ' |'] + ['| ' + ' | '.join(str(c).replace('<br/>', ' ') for c in row) + ' |' for row in values[1:]]))
    formatted = [[Paragraph(str(v), styles['Small']) for v in row] for row in values]
    t = Table(formatted, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e7eff2')), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#6c8792')), ('LINEBELOW', (0, -1), (-1, -1), .4, colors.HexColor('#b1c2c9'))]))
    story.extend([t, Spacer(1, 10)])
def fig(path, width, height, caption):
    story.append(Image(str(path), width=width, height=height)); p(caption, 'Small')
    md.append('![' + caption + '](../' + str(path.relative_to(ROOT)).replace('\\', '/') + ')')
def f(value, n=4): return f'{float(value):.{n}f}'
def pc(value): return f'{100*float(value):.2f}%'
def pm(row, metric): return f'{100*float(row[metric+"_mean"]):.2f} +/- {100*float(row[metric+"_std"]):.2f}'

def flow():
    d = Drawing(495, 325)
    def box(x, y, w, ht, lines, fill='#eef3f5'):
        d.add(Rect(x, y, w, ht, rx=4, ry=4, strokeColor=colors.HexColor('#6c8792'), fillColor=colors.HexColor(fill), strokeWidth=.6))
        for j, line in enumerate(lines): d.add(String(x+w/2, y+ht/2+(len(lines)-1)*6-j*12, line, textAnchor='middle', fontName='Helvetica', fontSize=9))
    def arrow(x1,y1,x2,y2):
        d.add(Line(x1,y1,x2,y2,strokeColor=colors.HexColor('#6c8792'),strokeWidth=.8))
        d.add(Polygon([x2,y2,x2-3,y2+6,x2+3,y2+6],fillColor=colors.HexColor('#6c8792'),strokeColor=None))
    box(72,275,350,42,['Align local and neighbor hourly weather data', 'Construct fixed next-hour Low / Medium / High labels'])
    arrow(247,275,247,260)
    box(72,220,350,40,['2024 development and 2025 future test', 'Same target rows across every feature arm'])
    for x, label in [(0,'Local current'),(126,'Temporal'),(252,'Spatial current'),(378,'Spatiotemporal')]:
        arrow(247,220,x+58,202);box(x,165,117,37,[label])
        arrow(x+58,165,247,147)
    box(72,105,350,42,['Matched stratified day-group validation in 2024', 'Training-only imputation and scaling'])
    arrow(247,105,247,90)
    box(72,48,350,42,['Fixed classical classifiers and validation-based selection', 'Refit on 2024, evaluate matched 2025 predictions'])
    arrow(247,48,247,35)
    box(72,0,350,35,['Metrics, paired differences, uncertainty and hypothesis verdict'])
    return d

a, b = test[('temporal', primary)], test[('spatiotemporal', primary)]
main_pair = next(r for r in paired if r['model'] == primary)
p('Temporal and Spatiotemporal Wind Regime Classification', 'Cover')
p('A classical pattern recognition study of low medium and high next hour wind speed', 'Heading2')
p('Semester project for Pattern Recognition<br/>Monsoon 2026<br/>Report date 5 November 2026')
sub('Research hypothesis')
p('Adding neighboring-station weather observations and their recent histories to local temporal features will improve next-hour wind-regime classification at Brentwood.')
sub('Abstract')
p(f'The supplied weather records are reformulated as a supervised three-class problem with fixed Low, Medium and High next-hour wind-speed labels. The experiment uses {info["instances"]:,} instances, ten raw meteorological attributes, and four matched feature conditions containing 10, 22, 28 and 46 attributes. Logistic regression, shrinkage linear discriminant analysis and a random forest form the main comparison. Five stratified calendar-day grouped folds on 2024 support model selection and mean plus sample-standard-deviation reporting. The complete 2025 year remains a future test set. The primary classifier, selected from temporal validation macro-F1, is {primary}. Temporal inputs achieve {pc(a["accuracy"])} accuracy and {f(a["macro_f1"],6)} macro-F1. Spatiotemporal inputs achieve {pc(b["accuracy"])} and {f(b["macro_f1"],6)}. Logistic regression and LDA show small test macro-F1 gains from spatial inputs, but the primary forest comparison does not improve. All paired day-bootstrap intervals include zero. The study therefore finds classifier-dependent, inconclusive spatial benefit rather than consistent superiority.')
sub('Repository and deliverable scope')
p('<link href="https://github.com/medhanshrao20/pr">https://github.com/medhanshrao20/pr</link>')
p('Main.m runs the complete new classification experiment and displays the required results. Original temporal and spatial forecasting code remains unchanged. This deliverable contains the report, methodology, executable code and computed evidence.')
page()

h('1 Problem formulation and dataset')
p('Short-term weather monitoring can benefit from a compact forecast of wind conditions. The real-world question here is whether other stations provide information beyond a target station\'s own recent observations. The target is a future category at Brentwood, not current wind reconstructed from itself. No power-generation or safety-warning benefit is inferred from these classification scores.')
tbl([['Dataset property','Value'],['Source','Supplied California CIMIS hourly station files'],['Target','Brentwood station 47'],['Candidate stations','14 including Brentwood'],['Raw target records','17,568'],['Usable classification instances',f'{info["instances"]:,}'],['2024 training / development',f'{info["train_n"]:,}'],['2025 future test',f'{info["test_n"]:,}'],['Raw attributes / target classes','10 / 3']], [190,305])
sub('Fixed category definitions')
tbl([['Category','Wind at t + 1 hour','2024 count','2025 count'],['Low','Below 3 mph',info['train_class_counts'][0],info['test_class_counts'][0]],['Medium','3 to below 7 mph',info['train_class_counts'][1],info['test_class_counts'][1]],['High','At least 7 mph',info['train_class_counts'][2],info['test_class_counts'][2]]],[80,195,110,110])
p('Thresholds are fixed project design choices, not an official meteorological warning scale. Labels are derived only from next-hour target wind. Current target wind may appear among the predictors because it is available one hour before the outcome. A label is accepted only when the target is observed, nonnegative, exactly one hour later, and on the same calendar day. Observations outside 2024 and 2025 are excluded.')
p('All arms use identical rows and outcomes. The raw target record count decreases by 762 after date and target-pair eligibility rules. Missing predictors and missing neighbors remain eligible and are imputed from training medians. The supplied quality-control columns are not predictors, and no exhaustive quality-flag filtering is claimed.')
page()

h('2 Temporal and spatial source lineage')
p('The existing repository contains a local temporal forecasting architecture and a target-only spatial adaptation. The temporal code was identified directly in the supplied tree. A separate classification implementation is added so that the course comparison focuses on classical pattern recognition and newly computed categorical outcomes.')
tbl([['Existing source','Role'],['L/run.py','Earlier local baseline with fixed seasonal mode counts and validation-based alpha selection.'],['L.1/run.py','Local temporal entry point for hybrid decomposition-parameter selection.'],['L.1/source/model.py','Original KANInformer temporal backbone.'],['spatial_L.1/run.py','Forecasting entry point for target-only spatial context.'],['common/graph_model.py','Graph adapter around the temporal backbone. Its Ridge pathway remains Brentwood-only.'],['common/experiment.py','Original new-data adapters, seasonal experiments and output management.']], [190,305])
sub('Classification adaptation')
p('The new temporal classifier consumes Brentwood measurements and past-only features. The new spatiotemporal classifier receives the same local representation plus geographically neighboring measurements and histories. Both use the same estimator settings, labels, data partitions and scoring functions. The added spatial computation is a deterministic geographic feature aggregation, not a newly trained graph neural network.')
tbl([['New entry point','Purpose'],['Main.m','Complete classification comparison and result display.'],['classification/run_experiment.py','All four feature conditions, matched classifiers and evidence generation.'],['classification/temporal_main.py','Independent temporal classification runner.'],['classification/spatiotemporal_main.py','Independent spatiotemporal classification runner.'],['classification/check_results.py','Metric, cohort, fold, causality and integrity checks.']], [245,250])
p('The 50 pre-existing source and data files are fingerprinted in docs/original_source_manifest.json and remain unchanged. Original continuous forecasting RMSE, MAE and MAPE in best_results.csv are retained as historical regression evidence. They cannot be converted into classification accuracy, macro-F1 or a confusion matrix. No old regression result is presented as a new categorical prediction.')
page()

h('3 Methodology flow and experimental logic')
story.append(flow()); md.append('See docs/METHODOLOGY.md for the editable Mermaid methodology flow.')
p('Figure 1. Classification methodology. Each feature arm uses the same target rows and paired validation/test observations. The primary classifier is selected from temporal validation results before test scoring.','Small')
sub('Directional hypothesis and primary comparison')
p('The research hypothesis predicts a positive spatiotemporal-minus-temporal macro-F1 difference. The working null is no increase. To avoid selecting whichever final result looks favorable, the classifier family is fixed using only the temporal arm\'s 2024 validation macro-F1. That same classifier then supplies the primary comparison between feature sets. Other classifiers provide secondary checks.')
sub('Ablation controls')
p('Local-current versus temporal evaluates extracted local histories and circular attributes. Local-current versus spatial-current evaluates contemporaneous neighbor information. Temporal versus spatiotemporal is the principal feature-package comparison. The spatiotemporal package adds both neighbor current measurements and neighbor histories, so this design cannot isolate each individual added feature or separate geographic weighting from the rest of the package.')
p('A more complex condition can help or hurt. Model-selection criteria, category thresholds, neighbors and classifier settings do not change after the test results become visible. Negative and inconclusive findings are retained.')
page()

h('4 Local and geographic feature construction')
tbl([['Condition','Attributes','Information'],['Local current','10','Current Brentwood weather'],['Temporal','22','Local current plus past and circular attributes'],['Spatial current','28','Local current plus neighbor current observations and summaries'],['Spatiotemporal','46','Temporal plus neighbor current observations, histories and summaries']],[110,65,320])
p('The ten raw attributes are evapotranspiration, precipitation, solar radiation, vapor pressure, air temperature, relative humidity, dew point, wind speed, wind direction and soil temperature. Units follow the supplied CSV. The sol_rad_ly_per_day label is retained without independently verifying the hourly semantics implied by its name.')
p('The temporal condition adds wind lags at one and three hours, trailing three-hour mean and sample SD, one-hour wind change, temperature minus dew point, and sine/cosine pairs for wind direction, hour of day and annual position. Circular encoding uses sin(theta) and cos(theta), with theta set by the corresponding period. Trailing windows use the current observation and earlier observations only. Histories reset at day boundaries; missing early-day histories are imputed within the training pipeline.')
tbl([['Neighbor','Distance from Brentwood','Geographic weight']]+[[n['name'],f'{n["distance_km"]:.2f} km',f(n['inverse_distance_weight'])] for n in info['neighbors']],[190,160,145])
p('Coordinates and haversine distance fix the three nearest neighbors. Weights are normalized inverse distances. Each neighbor supplies current wind, temperature, humidity and vapor pressure; the spatiotemporal arm adds its one-hour wind lag and trailing three-hour mean. For wind, temperature and humidity, weighted neighbor means and local-minus-neighbor differences add six summaries. Means renormalize over observed neighbors and remain missing if none are observed.')
fig(R/'neighbor_layout.png',440,225,'Figure 2. Station coordinate plot, without a basemap. Geographic separation motivates the fixed neighbor set but does not prove predictive dependence.')
page()

h('5 Classical models and leakage controls')
tbl([['Classifier','Fixed configuration'],['Logistic regression','L2 regularization, C=1, balanced class weights, maximum 2,000 iterations, seed 42.'],['Shrinkage LDA','lsqr solver with automatic covariance shrinkage.'],['Random forest','160 trees, depth 14, minimum leaf size 3, balanced subsample weights, default square-root feature sampling, seed 42.'],['Persistence','Predict the next-hour category from current Brentwood wind. Missing current wind uses the training median.']],[145,350])
p('Each learned classifier uses median imputation and standard scaling fitted on its training subset only. Tree models do not require scaling, but the same pipeline structure is retained. No full-season decomposition, backward fill, centered window, future neighbor value, supervised full-data feature selection or PCA enters the classification pipeline. The retained forecasting program\'s full-season preprocessing is not reused.')
sub('Training validation and test partitions')
p('Calendar year 2024 is the development pool. Five StratifiedGroupKFold partitions, with shuffle enabled and seed 42, approximately preserve class proportions while assigning each calendar day entirely to one fold. All folds contain every class. Every candidate uses the same folds. Within-day histories and next-hour labels therefore cannot overlap across training and validation groups. Days can still share weather regimes, so grouped validation is retrospective rather than a rolling-origin operational simulation.')
p('The full 2025 year forms a separate future holdout. Every candidate is refitted on all 2024 observations before final scoring. No test score controls selection, hyperparameters or neighbor choice. The classification study uses classical estimators only. The original deep temporal/spatial forecasting code provides source lineage and related context, not a directly comparable categorical benchmark.')
sub('Metrics and uncertainty')
p('Accuracy is the fraction of correct labels. For a class, precision = TP / (TP + FP), recall = TP / (TP + FN), and F1 = 2PR / (P + R). Macro-F1 averages the three class F1 scores equally. Undefined class ratios receive zero. Reported SD is the sample standard deviation across five validation folds (ddof=1), not a confidence interval. A paired calendar-day bootstrap of 2,000 resamples provides conditional percentile 95% intervals for final-test differences. It does not model across-day dependence or model-fitting uncertainty.')
page()

h('6 Five fold validation results')
tbl([['Condition','Classifier','Accuracy mean +/- SD','Macro-F1 mean +/- SD']]+[[names[c],m,pm(cv[(c,m)],'accuracy'),pm(cv[(c,m)],'macro_f1')] for c in conditions for m in models]+[['Persistence','Persistence',pm(cv[('persistence','Persistence')],'accuracy'),pm(cv[('persistence','Persistence')],'macro_f1')]],[105,100,145,145])
p('Table 1. Validation scores on a 0 to 100 scale, with sample SD in percentage points. Models use 2024 observations only. Same folds across all rows.','Small')
p(f'Temporal validation selects {primary}, whose mean macro-F1 is {f(cv[("temporal",primary)]["macro_f1_mean"])}. The same classifier is used in the primary spatiotemporal comparison. For that model, the paired validation macro-F1 change is {100*float(main_pair["cv_macro_f1_difference_mean"]):+.3f} +/- {100*float(main_pair["cv_macro_f1_difference_std"]):.3f} points. The mean of paired fold differences and its SD are computed from matched folds, rather than subtracting unrelated SDs.')
tbl([['Classifier','Paired accuracy change mean +/- SD','Paired macro-F1 change mean +/- SD']]+[[r['model'],f'{100*float(r["cv_accuracy_difference_mean"]):+.3f} +/- {100*float(r["cv_accuracy_difference_std"]):.3f}',f'{100*float(r["cv_macro_f1_difference_mean"]):+.3f} +/- {100*float(r["cv_macro_f1_difference_std"]):.3f}'] for r in paired],[105,195,195])
p('Table 2. Spatiotemporal minus temporal validation changes, on a 0 to 100 scale. Positive means the spatial feature package improves the score.','Small')
p('Validation supports trying spatial features, but validation gains do not establish gains on a later year. Keeping the future holdout separate is necessary to check that distinction.')
page()

h('7 Final future year classification results')
tbl([['Condition','Classifier','Attributes','Accuracy','Macro-F1']]+[[names[c],m,test[(c,m)]['features'],pc(test[(c,m)]['accuracy']),f(test[(c,m)]['macro_f1'],6)] for c in conditions for m in models]+[['Persistence','Persistence','10',pc(test[('persistence','Persistence')]['accuracy']),f(test[('persistence','Persistence')]['macro_f1'],6)]],[115,100,65,100,115])
p('Table 3. All candidates use the same 8,393 final-test instances from 2025. Six decimal places in macro-F1 preserve the very small primary-arm difference.','Small')
fig(R/'ablation_macro_f1.png',450,250,'Figure 3. Held-out macro-F1 for the four matched information conditions. Classifier settings are fixed across feature arms.')
p('The temporal random forest has the highest test accuracy among these fixed candidates. This observation does not change the prespecified classifier choice. The spatiotemporal condition improves the final-test macro-F1 of logistic regression and LDA, while the random-forest macro-F1 is slightly lower. More attributes alone do not guarantee improvement.')
page()

h('8 Primary comparison and class behavior')
imgs = [[Image(str(R/f'confusion_temporal_{primary}.png'),width=242,height=191), Image(str(R/f'confusion_spatiotemporal_{primary}.png'),width=242,height=191)]]
story.append(Table(imgs,colWidths=[247,248])); story.append(Spacer(1,10))
md.append(f'![Temporal confusion matrix](../results/classification/confusion_temporal_{primary}.png)\n\n![Spatiotemporal confusion matrix](../results/classification/confusion_spatiotemporal_{primary}.png)')
p('Figure 4. Temporal (left) and spatiotemporal (right) random-forest confusion matrices. Rows are true categories and columns are predictions. Both use exactly the same test outcomes.','Small')
tbl([['Arm and class','Precision','Recall','F1','N']]+[[names[r['condition']]+'<br/>'+r['class'],f(r['precision']),f(r['recall']),f(r['f1-score']),str(int(float(r['support'])))] for r in per if r['condition'] in ['temporal','spatiotemporal'] and r['model']==primary],[215,70,70,70,70])
p('The spatial package changes which categories benefit. Low-class recall falls from 0.8327 to 0.8142, while Medium recall rises from 0.7385 to 0.7478 and High recall rises from 0.8011 to 0.8082. These shifts nearly offset in macro-F1. Accuracy decreases by seven correct predictions out of 8,393, or about 0.083 percentage points. This is consistent with a small redistribution of errors rather than a clear overall benefit.')
p('Most confusions remain between adjacent categories. The feature thresholds create natural boundary cases, and Medium wind lies between two alternatives. Per-class precision and recall therefore remain necessary even when the headline macro-F1 values are almost equal.')
page()

h('9 Hypothesis assessment and uncertainty')
tbl([['Classifier','Test accuracy difference','Test macro-F1 difference','95% interval for macro-F1 difference']]+[[r['model'],f'{100*float(r["test_accuracy_difference"]):+.3f} points',f'{float(r["test_macro_f1_difference"]):+.6f}',f'[{float(r["macro_f1_difference_ci_low"]):+.6f}, {float(r["macro_f1_difference_ci_high"]):+.6f}]'] for r in paired],[100,110,115,170])
p('Table 4. Spatiotemporal minus temporal. Intervals use 2,000 paired calendar-day bootstrap resamples of 365 test days. Every interval includes zero.','Small')
fig(R/'paired_macro_f1.png',450,197,'Figure 5. Positive point estimates favor spatial information. Overlap with zero indicates that these conditional intervals do not establish a positive gain.')
sub('Primary verdict')
p(f'The prespecified primary comparison is {primary}. Its test macro-F1 difference is {float(main_pair["test_macro_f1_difference"]):+.6f}, with conditional 95% interval [{float(main_pair["macro_f1_difference_ci_low"]):+.6f}, {float(main_pair["macro_f1_difference_ci_high"]):+.6f}]. This experiment does not support consistent spatiotemporal superiority. The near-zero point difference is insufficient evidence that spatial information is harmful in general, just as the positive secondary differences are insufficient evidence of universal benefit.')
sub('Secondary observations')
p('Logistic regression and LDA obtain small positive test macro-F1 changes. Their intervals also cross zero. The direction differs by classifier and differs between validation and future-year evaluation for LDA. The result is therefore best described as classifier-dependent and inconclusive. A report that claimed spatial dominance from only the favorable rows would misrepresent the complete comparison.')
p('The bootstrap treats days as resampling blocks. It preserves within-day pairing but does not remove correlation between adjacent days. It is conditional on one fitted model per arm, one station and one year. These intervals are supporting diagnostics rather than a comprehensive significance claim.')
page()

h('10 Interpretation limitations and conclusion')
sub('Interpretation')
p('Local wind persistence is already strong. A neighboring measurement may be redundant with Brentwood\'s recent wind, or describe a different local weather regime. The selected stations are approximately 68 to 152 km from the target. Geographic closeness does not establish that their current readings lead the target by the relevant forecast horizon. Extra features also change the forest\'s randomly sampled feature subsets, so estimator behavior can vary even when spatial information is weak.')
p('The controls help separate information packages. Temporal features improve the raw random-forest condition, whereas adding only neighbor current observations yields little change in its test macro-F1. The combined condition has a small validation benefit without an overall future-test gain. These are descriptive effects of the complete feature packages, not causal claims about individual meteorological variables.')
sub('Limitations')
p('The study has one target station, a single future year, three geographically selected neighbors and fixed project category thresholds. Reported units follow the archived files. No comprehensive quality-flag cleaning, sensor latency study or deployment check is included. Neighbor features assume measurements at the current hour are available in time. Grouped folds can retain across-day weather dependence, and bootstrap intervals are conditional on the fitted models. No deep-learning classification comparator or original forecasting-model retraining was performed.')
p('The 2025 cohort was previously evaluated in an earlier local-only classification experiment. This is a fixed-protocol follow-up comparison, without retuning after its new spatial results, rather than a newly blind confirmatory test. An independent year or station is needed for stronger confirmation.')
sub('Conclusion')
p('The wind-speed forecasting context can be formulated as a valid multiclass pattern-recognition experiment while preserving existing source files. The reproducible comparison satisfies the dataset size, attribute count, class count and classical-method focus, and reports accuracy, macro-F1, all per-class measures, confusion matrices and validation mean plus SD. The tested spatial feature package does not establish consistent improvement over the temporal baseline. The value of the project is a fair, auditable test of that hypothesis, including its mixed outcome.')
sub('Future work')
p('Before an additional test, specify physically motivated lag alignment, candidate neighbor sets and category definitions using training/validation data only. Expand to other target stations and years, assess sensor availability, and use a nested chronological design. Additional comparisons should remain prespecified and retain negative results.')
page()

h('11 References and reproducibility')
p('[1] California Department of Water Resources. California Irrigation Management Information System. https://cimis.water.ca.gov/. Weather report resources: https://cimis-proxy.water.ca.gov/Resources.aspx. Accessed 5 November 2026.', 'Small')
p('[2] scikit-learn developers. Cross-validation: evaluating estimator performance. https://scikit-learn.org/stable/modules/cross_validation.html. Grouped and stratified validation guidance. The executed environment uses scikit-learn 1.3.2.', 'Small')
p('[3] scikit-learn developers. Common pitfalls and recommended practices. https://scikit-learn.org/stable/common_pitfalls.html. Training-only transformations and pipelines.', 'Small')
p('[4] Leng Z, Chen L, Yi B, Liu F, Xie T, Mei Z. Short-term wind speed forecasting based on a novel KANInformer model and improved dual decomposition. Energy 322, 135551 (2025). https://doi.org/10.1016/j.energy.2025.135551. Related forecasting work cited by the supplied manuscript, not a classification benchmark.', 'Small')
sub('Run and verify')
p('Clone https://github.com/medhanshrao20/pr. In MATLAB, run Main.m to generate every classification arm, score table, prediction table and confusion figure. Python-only execution: python -B classification/run_experiment.py. Verification: python -B classification/check_results.py. Separate temporal and spatiotemporal entry points are included for inspection and isolated reruns. They use separate output folders.')
p('Install numerical dependencies from requirements.txt. MATLAB automatically attempts installation if required Python imports are missing. Report regeneration additionally requires requirements-report.txt, then python tools/build_figures.py and python tools/build_report.py. Report generation consumes saved evidence and does not fit or tune models. The original forecasting dependencies are isolated in requirements-forecasting.txt and are not needed for the new classical experiment.')
tbl([['Runtime item','Recorded version']]+[[k,v] for k,v in info['versions'].items()]+[['Random seed','42'],['Validation folds','5'],['Bootstrap resamples','2,000']],[220,275])
p('The dataset, coordinate and original-source SHA-256 hashes are preserved in the input audit and source manifest. Saved test and validation predictions permit recomputation of metrics. split_assignments.csv records timestamp-level partitions. Keep the repository accessible until course grades are awarded. No perpetual hosting guarantee or third-party material license is inferred.')
sub('Appendix A Individual contributions')
p('Team members must supply and verify their names, roll numbers and actual contributions before submission. No individual authorship or work allocation is inferred from repository ownership. Complete CONTRIBUTIONS.md and update this appendix with the verified information.')
page()

for index, model in enumerate(models, 1):
    h('Appendix B' + str(index) + ' All per class results for ' + model)
    tbl([['Condition and category','Precision','Recall','F1','N']]+[[names[r['condition']]+'<br/>'+r['class'],f(r['precision']),f(r['recall']),f(r['f1-score']),str(int(float(r['support'])))] for r in per if r['model']==model],[215,70,70,70,70])
    p('All rows use the same held-out 2025 test instances. The category counts are fixed across feature conditions. Corresponding full confusion matrices and per-instance predictions are available in results/classification.', 'Small')
    if index == 3:
        sub('Persistence baseline')
        tbl([['Category','Precision','Recall','F1','N']]+[[r['class'],f(r['precision']),f(r['recall']),f(r['f1-score']),str(int(float(r['support'])))] for r in per if r['model']=='Persistence'],[215,70,70,70,70])
    if index < 3: page()

def footer(canvas, doc):
    canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.HexColor('#536d78'))
    canvas.drawRightString(545, 25, str(doc.page))
pdf = OUT / 'Temporal_vs_Spatiotemporal_Report.pdf'
SimpleDocTemplate(str(pdf), pagesize=(595.28,841.89), leftMargin=50, rightMargin=50, topMargin=43, bottomMargin=43,
    title='Temporal and Spatiotemporal Wind Regime Classification', author='').build(story,onFirstPage=footer,onLaterPages=footer)
(OUT / 'Temporal_vs_Spatiotemporal_Report.md').write_text('\n\n'.join(md),encoding='utf-8')
print(pdf)
