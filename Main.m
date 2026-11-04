% Run the complete classical classification experiment without interactive input.
% Requires MATLAB and Python 3.10+ on PATH. Original forecasting sources are unchanged.
root = fileparts(mfilename('fullpath'));
pythonCommand = 'python';
[status, ~] = system([pythonCommand ' --version']);
if status ~= 0
    pythonCommand = 'py -3';
end
[status, ~] = system([pythonCommand ' -c "import numpy,pandas,sklearn,matplotlib"']);
if status ~= 0
    command = sprintf('%s -m pip install -r "%s"', pythonCommand, fullfile(root, 'classification', 'requirements.txt'));
    [status, output] = system(command, '-echo');
    assert(status == 0, 'Dependency installation failed: %s', output);
end
command = sprintf('%s -B "%s"', pythonCommand, fullfile(root, 'classification', 'run_classification.py'));
[status, output] = system(command, '-echo');
assert(status == 0, 'Classification experiment failed: %s', output);
images = dir(fullfile(root, 'results', 'classification', 'confusion_*.png'));
for k = 1:numel(images)
    figure('Name', images(k).name, 'NumberTitle', 'off');
    image(imread(fullfile(images(k).folder, images(k).name)));
    axis image off;
end
fprintf('\nResults saved in %s\n', fullfile(root, 'results', 'classification'));
