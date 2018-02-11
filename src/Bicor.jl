__precompile__(true)

module Bicor
    using Missings

    """
    General notes about handling missing data, zero MAD etc:
    The idea is that bicor should switch to cor whenever it is feasible, it helps, and it is requested:
    (1) if median is NA, the mean would be NA as well, so there's no point in switching to Pearson
    (2) In the results, columns and rows corresponding to input with NA means/medians are NA'd out.
    (3) The convention is that if zeroMAD is set to non-zero, it is the index of the column in which MAD is
        zero (plus one for C indexing)
    """

    const RefUX = 0.5


    func2(x, ::Type{Val{true}}, lowQ, hiQ) = x * RefUX / lowQ
    func2(x, ::Type{Val{false}}, lowQ, hiQ) = x * RefUX / hiQ
    func2(x, ::Type{Val{missing}}, lowQ, hiQ) = missing
    func2(x, y, lowQ, hiQ) = func2(x, Val{y}, lowQ, hiQ)
    func1(x, lowQ, hiQ) = func2(x, x < 0, lowQ, hiQ)

    function prepareColBicor(col::Vector, maxPOutliers::Float64, fallback::Int64, cosine::Bool)
        """
        const double asymptCorr = 1.4826, qnorm75 = 0.6744898;
        Note to self: asymptCorr * qnorm75 is very close to 1 and should equal 1 theoretically. Should
        probably leave them out completely.
        """
        nr = length(col)
        clean_col = collect(AbstractFloat, skipmissing(col))
        nNAentries::UInt64 = nr - length(clean_col)
        NAmed::Bool = false
        zeroMAD::Bool = false
        res = copy(col)
        aux = copy(col)

        if fallback == 4
            res, nNAentries, NAmed = prepareColCor(col, cosine)
            return (res, nNAentries, NAmed, zeroMAD)
        end

        # Calculate the median of col
        med = median(clean_col)  # original one implicitly arrange NA to the end

        # Create a conditional copy of the median
        medX = (cosine)? 0.0 : med

        # calculate absolute deviations from the median
        zeroMAD = false
        NAmed = ismissing(med)
        if NAmed
            res = zeros(nr)
            zeroMAD = true
        else
            res = col .- medX
            aux = abs.(col .- med)

            # calculate mad, i.e. median absolute deviation
            mad = median(collect(AbstractFloat, skipmissing(aux)))  # original one implicitly arrange NA to the end

            # If mad is zero, value of fallback decides what is it we will do.
            if mad == 0.0
                zeroMAD = true
                if fallback == 1
                    # Return after zeoring out results and setting the NAmed flag
                    res = zeros(nr)
                    NAmed = true
                elseif fallback == 2
                    # Switch to Pearson correlation and return
                    # Rprintf("mad is zero in a column. Switching to Pearson for this column.\n");
                    res, nNAentries, NAmed = prepareColCor(col, cosine)
                elseif fallback == 3
                    # Do nothing: the setting of *zeroMAD above is enough.
                end
                return res, nNAentries, NAmed, zeroMAD
            end

            # We now re-use aux to store a copy of the weights ux. To calculate them, first get (x-med)/(9*mad).
            # Rprintf("median: %6.4f, mad: %6.4f, cosine: %d\n", med, mad, cosine);
            aux = (col - med) / (9.0 * mad)

            # Get the low and high quantiles  of ux
            clean_aux = collect(AbstractFloat, skipmissing(aux))
            lowQ = quantile(clean_aux, maxPOutliers)
            hiQ = quantile(clean_aux, 1.0 - maxPOutliers)

            # Rprintf("prepareColBicor: lowQ=%f, hiQ = %f\n", lowQ, hiQ);
            # If the low quantile is below -1, rescale the aux (that serve as ux below)
            # such that the low quantile will fall at -1; similarly for the high quantile

            (lowQ > -RefUX) && (lowQ = -RefUX)
            (hiQ < RefUX) && (hiQ = RefUX)
            lowQ = abs(lowQ)
            aux = func1.(aux, lowQ, hiQ)

            # Calculate the (1-ux^2)^2 * (x-median(x))
            sum::BigFloat = BigFloat(0.0)
            for k=1:nr
                if !ismissing(res[k])
                    ux = (abs(aux[k]) > 1)? 1.0 : aux[k]  # sign of ux doesn't matter.
                    sum += (res[k] * (1 - ux^2)^2)^2
                else
                    res[k] = 0
                end
            end
            sum = √(sum)
            if sum == 0.0
                res = zeros(nr)
                NAmed = true
            else
                res = Float64.(res ./ sum)
            end
        end
        return res, nNAentries, NAmed, zeroMAD
    end

    function prepareColCor(x::Vector, cosine::Bool)
        nr = length(x)
        # res
        nNAentries::UInt64 = 0
        NAmean::Bool

        count::UInt64 = 0
        mean::BigFloat = BigFloat(0.0)
        sum::BigFloat = BigFloat(0.0)
        for k=1:nr
            if !ismissing(x[k])
                mean += x[k]
                sum += x[k]^2
                count += 1
            end
        end

        if count > 0
            NAmean = false
            nNAentries = nr - count
            mean = (cosine)? 0 : mean/count
            sum = √(sum - count * mean*mean)
            if sum > 0
                # Rprintf("sum: %Le\n", sum)
                for k=1:nr
                    res[k] = (!ismissing(x[k]))? (x[k] - mean)/sum : 0.0
                end
            else
                # Rprintf("prepareColCor: have zero variance.\n")
                NAmean = true
                res = zeros(nr)
            end
        else
            NAmean = true
            nNAentries = nr
            res = zeros(nr)
        end
        return res, nNAentries, NAmean
    end

    export prepareColBicor;


    # Interfaces
    # """
    #  Pearson correlation of a matrix with itself.
    #  This one uses matrix multiplication in BLAS to speed up calculation when there are no NA's
    #  and uses threading to speed up the rest of the calculation.
    # """
    # void cor1Fast(double * x, int * nrow, int * ncol, double * quick,
    #           int * cosine,
    #           double * result, int *nNA, int * err,
    #           int * nThreads,
    #           int * verbose, int * indent);

    # """
    # bicorrelation of a matrix with itself.
    # This one uses matrix multiplication in BLAS to speed up calculation when there are no NA's
    # and is threaded to speed up the rest of the calculation.
    # """
    # void bicor1Fast(double * x, int * nrow, int * ncol,
    #             double * maxPOutliers, double * quick,
    #             int * fallback, int * cosine,
    #             double * result, int *nNA, int * err,
    #             int * warn,
    #             int * nThreads,
    #             int * verbose, int * indent);

    # """
    # Two-variable bicorrelation. Basically the same as bicor1, just must calculate the whole matrix.
    # If robustX,Y is zero, the corresponding variable will be treated as in pearson correlation.
    # """
    # void bicorFast(double * x, int * nrow, int * ncolx, double * y, int * ncoly,
    #            int * robustX, int * robustY,
    #            double * maxPOutliers, double * quick,
    #            int * fallback,
    #            int * cosineX, int * cosineY,
    #            double * result, int *nNA, int * err,
    #            int * warnX, int * warnY,
    #            int * nThreads,
    #            int * verbose, int * indent);

    # """
    # This can actually be relatively slow, since the search for calculations that need to be done is not
    # parallel, so one thread may have to traverse the whole matrix. I can imagine parallelizing even that
    # part, but for now leave it as is as this will at best be a minuscule improvement.
    # """
    # void corFast(double * x, int * nrow, int * ncolx, double * y, int * ncoly,
    #            double * quick,
    #            int * cosineX, int * cosineY,
    #            double * result, int *nNA, int * err,
    #            int * nThreads,
    #            int * verbose, int * indent);

end # module
