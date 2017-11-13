function [measure_mean, selected_idx, varargout] = multi_label_eval(label_vs_candidate, varargin)
    %% Assert arguments are correct
    assert(ismatrix(label_vs_candidate), 'label_vs_candidate must be a 2-dimensional matrix');
    
    %% Parse varargins
    threshold = [];
    for i = 1:length(varargin)
        if (strcmp(varargin{i}, 'threshold'))
            assert(length(varargin) > i, 'missing value for option ''threshold''');
            assert(isnumeric(varargin{i+1}), 'value for option ''threshold'' must be numeric');
            threshold = varargin{i+1};
        end
    end
    
    %% Invert matrix since munkres searches for the minimal cost
    max_value               = max(max(label_vs_candidate));
    label_vs_candidate_inv  = max_value - label_vs_candidate;
    
    %% Find linear assignment
    [assignment, ~] = munkres(label_vs_candidate_inv);
    
    %% Re-compute mean since munkres will find the minimum but we want to find the maximum in the given input
    candidate_idx                   = assignment;                         % temp store for assignments
    candidate_idx(assignment < 1)   = NaN;                                % removes 0 values (needed for sub2ind)
    n_labels                        = size(label_vs_candidate, 1);
    selected_idx                    = sub2ind(size(label_vs_candidate), 1:n_labels, candidate_idx);
    selected_idx                    = selected_idx(~isnan(selected_idx)); % remove NaN values (needed for logical index)
    selected_values                 = label_vs_candidate(selected_idx);   % select assigned values
    measure_mean                    = sum(selected_values) / n_labels;    % compute mean
    
    %% Compute number of values above threshold
    if (~isempty(threshold))
        varargout{1} = sum(selected_values >= threshold);
    end
end