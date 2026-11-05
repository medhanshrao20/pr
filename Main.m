% Run all temporal versus spatiotemporal classical classification comparisons.
root = fileparts(mfilename('fullpath'));
pythonCommand = 'python';
[status, ~] = system([pythonCommand ' --version']);
if status ~= 0
    pythonCommand = 'py -3';
end
[status, ~] = system([pythonCommand ' -c "import numpy,pandas,scipy,sklearn,matplotlib"']);
if status ~= 0
    [status, output] = system(sprintf('%s -m pip install -r "%s"', pythonCommand, fullfile(root, 'requirements.txt')), '-echo');
    assert(status == 0, 'Python dependency installation failed: %s', output);
end
[status, output] = system(sprintf('%s -B "%s"', pythonCommand, fullfile(root, 'classification', 'run_experiment.py')), '-echo');
assert(status == 0, 'Classification comparison failed: %s', output);
resultRoot = fullfile(root, 'results', 'classification');
files = dir(fullfile(resultRoot, 'confusion_*.png'));
files(end + 1).name = 'ablation_macro_f1.png';
for k = 1:numel(files)
    figure('Name', files(k).name, 'NumberTitle', 'off');
    image(imread(fullfile(resultRoot, files(k).name)));
    axis image off;
end
fprintf('\nAll classification tables and prediction evidence: %s\n', resultRoot);
