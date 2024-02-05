data = load("data/data.mat").input_data;
datasets = length(data);
bounds = [50 * 1e-3, 50 * 1e-3, 50 * 1e-3, 60 * pi / 180, 60 * pi / 180, 60 * pi / 180];
nvars = 6;

all_times = zeros(5,3);

repeats = 10;
times_F = zeros(datasets,repeats);
times_dF = zeros(datasets,repeats);
times_ddF = zeros(datasets,repeats);

% common parametes
ii = 0;
for size_chunk = [1 10 100 1000 10000 100000]
    ii = ii+1;
    for i = 1:datasets
        fprintf("set %d:", i);
        F = @(x) valF(x, data{i}.S, data{i}.f_presc, data{i}.v, data{i}.n, data{i}.L);
        dF = @(x) gradF(x, data{i}.S, data{i}.f_presc, data{i}.v, data{i}.n, data{i}.L);
        ddF = @(x) hessF(x, data{i}.S, data{i}.f_presc, data{i}.v, data{i}.n, data{i}.L);
        x = 2 * (rand(size_chunk, nvars) - 0.5) .* bounds;
        for j = 1:repeats
            tic;
            F(x);
            time = toc;
            times_F(i, j) = time;

            tic;
            dF(x);
            time = toc;
            times_dF(i, j) = time;

            tic;
            ddF(x);
            time = toc;
            times_ddF(i, j) = time;

            fprintf(".");
        end
        fprintf("\n");
    end

all_times(ii,1) = median(times_F(:))/size_chunk;
all_times(ii,2) = median(times_dF(:))/size_chunk;
all_times(ii,3) = median(times_ddF(:))/size_chunk;
end
