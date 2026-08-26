classdef twodi_NetworkLRMLP_Class < handle
    % Locally Recurrent Multi-Layer Perception

    properties (Access = protected)
        t, D, M, R, S, nrOfPrms, f, df % Defining parameters of the network.
        n, a                           % Outputs. Initially they will be NaN.
        da_dx                          % Derivatives.
        IW, LW, b                      % Parameters.
    end

    methods % Constructor
        
        function obj = twodi_NetworkLRMLP_Class(options)

            arguments

                options.nrOfLayers  double {mustBeInteger,mustBePositive} = []
                options.nrOfInputs  double {mustBeInteger,mustBePositive} = []
                options.nrOfNeurons double {mustBeInteger,mustBePositive} = []
                options.actvnFcn    cell = {}
                options.dActvnFcn   cell = {}
       
            end

            % Defining parameters of the network
            obj.M        = options.nrOfLayers;
            obj.R        = options.nrOfInputs;
            obj.S        = cell(obj.M, 1);         for m = 1 : 1 : obj.M, obj.S{m}     = options.nrOfNeurons(m);                       end
            obj.nrOfPrms = obj.S{1} * (obj.R + 1); for m = 2 : 1 : obj.M, obj.nrOfPrms = obj.nrOfPrms + obj.S{m} * (obj.S{m - 1} + 1); end
            obj.f        = options.actvnFcn;
            obj.df       = options.dActvnFcn;
            obj.D        = 1;
            obj.t        = obj.D + 1;

            % Initialize outputs and their delays with zeros
            obj.n  = cell(obj.M, 1); obj.a  = cell(obj.M, 1); 
            for m = 1 : 1 : obj.M
                obj.a{m} = cell(obj.D + 1, 1); obj.n{m} = cell(obj.D + 1, 1);

                obj.a{m}{obj.t} = NaN(obj.S{m}, 1); obj.n{m}{obj.t} = NaN(obj.S{m}, 1);
                for d = 1 : 1 : (obj.D + 1)
                    obj.a{m}{d} = zeros(obj.S{m}, 1); obj.n{m}{d} = zeros(obj.S{m}, 1);
                end
            end

            % Initialize derivatives with zeros
            obj.da_dx = cell(obj.M, 1);
            for m = 1 : 1 : obj.M
                obj.da_dx{m} = cell(obj.D + 1, 1);

                for d = 1 : 1 : (obj.D + 1)
                    obj.da_dx{m}{d} = zeros(obj.S{m}, obj.getNrOfPrms);
                end
            end

            % Initialize weights with random, small values
            obj.IW = cell(obj.M, 1); obj.LW = cell(obj.M, obj.M); obj.b  = cell(obj.M, 1);
            for m = 1 : 1 : obj.M
                if m == 1
                    obj.IW{m} = 0.1 * randn(obj.S{m}, obj.R);
                else
                    obj.IW{m} = [];
                end
                for mm = 1 : 1 : obj.M
                    if((m == 1 && mm == obj.M) || (m == mm + 1))
                        obj.LW{m, mm} = 0.1 * randn(obj.S{m}, obj.S{mm});
                    else
                        obj.LW{m, mm} = [];
                    end
                end
                obj.b{m} = 0.1 * randn(obj.S{m}, 1);
            end
        end
    end

    methods

        function [n, a] = evaluate(obj, input)

            da_dx = obj.da_dx; n = obj.n; a = obj.a; IW = obj.IW; LW = obj.LW; b = obj.b; f = obj.f; t = obj.t;

            % Update delays
            for m = 1 : 1 : obj.M
                for d = 1 : 1 : obj.D
                    a{m}{d}     = a{m}{d + 1};
                    n{m}{d}     = n{m}{d + 1};
                    da_dx{m}{d} = da_dx{m}{d+1};
                end
            end

            % Update the outputs
            for m = 1 : 1 : obj.M
                if(m == 1)
                    n{m}{t} = IW{m, 1} * input + LW{1, obj.M} * a{obj.M}{t - 1} + b{m};
                else
                    n{m}{t} =                    LW{m, m - 1} * a{m - 1}{t    } + b{m};
                end
                a{m}{t} = f{m}(n{m}{t});
            end

            obj.a = a; obj.n = n; obj.da_dx = da_dx;
        end

        function daM_dx = fwdSnvty_daM_dx(obj, input)

            da_dx = obj.da_dx; n = obj.n; a = obj.a; LW = obj.LW; df = obj.df; t = obj.t; M = obj.M; 

            da_dx{1}{t} = diag(df{1}(n{1}{t}))                                                          * ...
                                (                                                                         ...
                                  squeeze(pagemtimes(obj.dIW_dx(    ), reshape(input      , [], 1, 1))) + ...
                                  squeeze(pagemtimes(obj.dLW_dx(1, M), reshape(a{M}{t - 1}, [], 1, 1))) + ...
                                  obj.db_dx(1)                                                          + ...
                                  LW{1, M} * da_dx{M}{t - 1}                                              ...
                                );

            for m = 2 : 1 : M
                da_dx{m}{t} = diag(df{m}(n{m}{t}))                                                              * ...
                                    (                                                                             ...
                                      squeeze(pagemtimes(obj.dLW_dx(m, m - 1), reshape(a{m - 1}{t}, [], 1, 1))) + ...
                                      LW{m, m - 1} * da_dx{m - 1}{t}                                            + ...
                                      obj.db_dx(m)                                                                ...
                                    );
            end
            
            daM_dx    = da_dx{M}{t};
            obj.da_dx = da_dx;
        end

        function dF_daM = backwSnvty_dF_daM(obj, measurement)

            a = obj.a; t = obj.t; M = obj.M; 

            dF_daM = -2 * (measurement - a{M}{t})';

        end

    end

    methods % Setters

        function setPrms(obj, x)

            b = obj.b; IW = obj.IW; LW = obj.LW; M = obj.M; 

            % Layer  1
            x1 = x(1 : obj.getNrOfPrmsInLayer(1));

            IW{1}    = reshape(x1(               1 : numel(IW{1}))                  , size(IW{1   }));
            LW{1, M} = reshape(x1(numel(IW{1}) + 1 : numel(IW{1}) + numel(LW{1, M})), size(LW{1, M}));
            b{1}     = x1(numel(IW{1}) + numel(LW{1, M}) + 1 : end);

            % Layers 2 ... M
            for m = 2 : 1 : M
        
                xm = x(obj.getNrOfPrmsUpToIncludingLayer(m - 1) + 1 : obj.getNrOfPrmsUpToIncludingLayer(m));

                LW{m, m - 1} = reshape(xm(1 : numel(LW{m, m - 1})), size(LW{m, m - 1}));
                b{m}         = xm(numel(LW{m, m - 1}) + 1 : end);
        
            end

            obj.b = b; obj.IW = IW; obj.LW = LW;
        end

    end

    methods % Getters

        function M = getNrOfLayers(obj), M = obj.M; end
        function D = getNrOfDelays(obj), D = obj.D; end

        function out = getNrOfPrmsInLayer (obj, m)

            IW = obj.IW; LW = obj.LW; b = obj.b; M = obj.M; %#ok<*PROPLC>
            if(m == 1)
                out = numel(IW{1}) + numel(LW{1, M}) + numel(b{1});
            else
                out = numel(LW{m, m - 1}) + numel(b{m});
            end
        end

        function out = getNrOfPrmsUpToIncludingLayer(obj, m)
            out = 0;
            for layer = 1 : 1 : m
                out = out + obj.getNrOfPrmsInLayer(layer);
            end
        end

        % x = [vec(IW{1}); vec(LW{1,M}); b{1}; vec(LW{2, 1}); b{2}; ...; vec(LW{M, M-1}); b{M}]
        function x = getPrms(obj)

            x = NaN(obj.getNrOfPrms, 1);
            for m = 1 : 1 : obj.M
        
                idxStart = obj.getNrOfPrmsUpToIncludingLayer(m - 1) + 1;
                idxEnd   = obj.getNrOfPrmsUpToIncludingLayer(m);
        
                if(m == 1)
                    x(idxStart : idxEnd, 1) = [obj.IW{m}(:); obj.LW{1, obj.M}(:); obj.b{m}];
                else
                    x(idxStart : idxEnd, 1) = [              obj.LW{m, m - 1}(:); obj.b{m}];
                end
            end
            if any(isnan(x), 'all'), error('NaN detected. Internal error.'); end
        end

        function nrOfPrms = getNrOfPrms(obj)

            nrOfPrms = obj.S{1} * obj.R + obj.S{1};
            nrOfPrms = nrOfPrms + obj.S{1} * obj.S{obj.M};

            for m = 2 : 1 : obj.M
                nrOfPrms = nrOfPrms + obj.S{m} * obj.S{m - 1} + obj.S{m};
            end
        end

    end

    methods (Access = private)

        function dIW_dx = dIW_dx(obj) % Now we are talking about tensors of order >= 3
            
            dIW_dx = zeros(obj.S{1}, obj.R, obj.getNrOfPrms());
            for k = 1 : 1 : numel(obj.IW{1})
                E = zeros(size(obj.IW{1})); E(k) = 1; % MATLAB linear indexing follows vec(IW)
                dIW_dx(:, :, k) = E;
            end
        end

        function dLW_dx = dLW_dx(obj, toLayer, fromLayer)

            if((toLayer == 1) && (fromLayer == obj.M))

                idxStart = numel(obj.IW{1}) + 1;
                dLW_dx   = zeros(obj.S{1}, obj.S{obj.M}, obj.getNrOfPrms());

            elseif(toLayer == (fromLayer + 1))

                idxStart = obj.getNrOfPrmsUpToIncludingLayer(fromLayer) + 1;
                dLW_dx   = zeros(obj.S{toLayer}, obj.S{fromLayer}, obj.getNrOfPrms());

            end

            for k = 1 : 1 : numel(obj.LW{toLayer, fromLayer})
                E = zeros(size(obj.LW{toLayer, fromLayer})); E(k) = 1;
                dLW_dx(:, :, idxStart + k - 1) = E;
            end

        end

        function db_dx = db_dx(obj, layer)

            db_dx = zeros(obj.S{layer}, obj.getNrOfPrms());

            if layer == 1
                idxStart = numel(obj.IW{1}) + numel(obj.LW{1,obj.M}) + 1;
            else
                idxStart = obj.getNrOfPrmsUpToIncludingLayer(layer) - numel(obj.b{layer}) + 1;
            end
        
            for k = 1:numel(obj.b{layer})
                db_dx(k, idxStart + k - 1) = 1;
            end

        end

    end

end