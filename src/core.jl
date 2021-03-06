# Interfaces
function cor(x::AbstractArray, cosine::Bool, verbose::Bool)::Void
    """
    Pearson correlation of a matrix with itself.
    This one uses matrix multiplication in BLAS to speed up calculation when there are no NA's
    and uses threading to speed up the rest of the calculation.
    """
    double * result, int *nNA
    nr, nc = size(x)
    nNA_ext = 0

    # This matrix will hold preprocessed entries that can be simply multiplied together to get the
    # numerator
    multMat = zeros(nr, nc)
    # Number of NA entries in each column
    nNAentries = zeros(nc)
    # Flag indicating whether the mean of each column is NA
    NAmean = falses(nc)

    # Decide how many threads to use
    nt = useNThreads( nc*nc, *nThreads)

    # Put the general data of the correlation calculation into a structure that can be passed on to
    # threads.
    cor1ThreadData thrdInfo[MxThreads];  # initailized

    # Column preparation (calculation of the matrix to be multiplied) in a threaded form.
    cptd = Array{colPrepThreadData}(MxThreads)
    thr = Array{pthread_t}(MxThreads)
    status = Array{Int64}(MxThreads)  # normal: 0
    pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
    progressCounter pc; pc.i = 0; pc.n = nc;

    # calculate threadPrepColCor using data cptd in each thr

    result = multMat' * multMat  # set NA means to zero

    # Here I need to recalculate results that have NA's in them.

    nNA = (int) nNA_ext;
    free(NAmean);
    free(nNAentries);
    free(multMat);
end

function bicor(x::AbstractArray, maxPOutliers::Float64, fallback::Int64, cosine::Bool,
               verbose::Bool)::Void
    """
    bicorrelation of a matrix with itself.
    This one uses matrix multiplication in BLAS to speed up calculation when there are no NA's
    and is threaded to speed up the rest of the calculation.
    """
    double * result, int *nNA
    nr, nc = size(x)
    nNA = 0
    nNA_ext = 0

    # This matrix will hold preprocessed entries that can be simply multiplied together to get the
    # numerator
    multMat = zeros(nr, nc)
    # Number of NA entries in each column
    nNAentries = zeros(nc)
    # Flag indicating whether the mean of each column is NA
    NAmed = falses(nc)

    # Decide how many threads to use
    int nt = useNThreads( nc*nc, *nThreads);

    double * aux[MxThreads];

    for (int t=0; t < nt; t++)
    {
       if ( (aux[t] = (double *) malloc(6*nr * sizeof(double)))==NULL)
       {
         *err = 1;
         Rprintf("cor1: memory allocation error. The needed block is very small... suspicious.\n");
         for (int tt = t-1; tt>=0; tt--) free(aux[tt]);
         free(NAmed); free(nNAentries); free(multMat);
         return;
       }
    }

    # Put the general data of the correlation calculation into a structure that can be passed on to
    # threads.

    cor1ThreadData thrdInfo[MxThreads];

    # Column preparation (calculation of the matrix to be multiplied) in a threaded form.
    cptd = Array{colPrepThreadData}(MxThreads)
    thr = Array{pthread_t}(MxThreads)
    status = Array{Int64}(MxThreads)  # normal: 0
    pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
    progressCounter pc; pc.i = 0; pc.n = nc;

    # Rprintf("Preparing columns...\n");
    # calculate threadPrepColBicor using data cptd in each thr

    int pearson = 0;

    if (*fallback==3)
    {
      for (int t=0; t<nt; t++) if (thrdInfo[t].zeroMAD > 0)
      {
        pearson = 1;
        if (*verbose)
          Rprintf("Warning in bicor(x): Thread %d (of %d) reported zero MAD in column %d. %s",
                  t, nt, thrdInfo[t].zeroMAD, "Switching to Pearson correlation.\n");
      }
      if (pearson==1) # Re-do all column preparations using Pearson preparation.
      {
        # Set fallback to 4 for slow calculations below.
        for (int t = 0; t < nt; t++) thrdInfo[t].fallback = 4;

        pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;
        pc.i = 0; pc.n = nc;
        # calculate threadPrepColCor using data cptd in each thr with thrdInfo
      }
    }

    result = multMat' * multMat  # lower triangle only

    # Here I need to recalculate results that have NA's in them.

    # Symmetrize the result and set all rows and columns with NA means to zero

    for (int t=nt-1; t >= 0; t--) free(aux[t]);

    *nNA = (int) nNA_ext;

    free(NAmed);
    free(nNAentries);
    free(multMat);
end

function bicor(x::AbstractArray, y::AbstractArray, robustX::Bool, robustY::Bool,
               maxPOutliers::Float64, fallback::Int64, cosineX::Bool, cosineY::Bool,
               verbose::Bool)::Void
    """
    Two-variable bicorrelation. Basically the same as bicor1, just must calculate the whole matrix.
    If robustX,Y is zero, the corresponding variable will be treated as in pearson correlation.
    """
    double * result, int *nNA
    assert(size(x, 1) == size(y, 1))
    nr = size(x, 1)
    ncx = size(x, 2)
    ncy = size(y, 2)
    nNA_ext = 0

    # This matrix will hold preprocessed entries that can be simply multiplied together to get the
    # numerator
    multMatX = zeros(nr, ncx)
    multMatY = zeros(nr, ncy)
    # Number of NA entries in each column
    nNAentriesX = zeros(ncx)
    nNAentriesY = zeros(ncy)
    # Flag indicating whether the mean of each column is NA
    NAmedX = falses(ncx)
    NAmedY = falses(ncy)

    # Decide how many threads to use
    int nt = useNThreads( ncx* ncy, *nThreads);

    double * aux[MxThreads];
    for (int t=0; t < nt; t++)
    {
       if ( (aux[t] = (double *) malloc(6*nr * sizeof(double)))==NULL)
       {
         *err = 1;
         Rprintf("cor1: memory allocation error. The needed block is very small... suspicious.\n");
         for (int tt = t-1; tt>=0; tt--) free(aux[tt]);
         free(NAmedY); free(NAmedX); free(nNAentriesY); free(nNAentriesX); free(multMatY); free(multMatX);
         return;
       }
    }

    cor1ThreadData thrdInfoX[MxThreads];
    cor1ThreadData thrdInfoY[MxThreads];
    cor2ThreadData thrdInfo[MxThreads];

    # Prepare the multMat columns in X and Y

    # Rprintf(" ..preparing columns in x\n");
    cptd = Array{colPrepThreadData}(MxThreads)
    thr = Array{pthread_t}(MxThreads)
    status = Array{Int64}(MxThreads)  # normal: 0

    progressCounter pcX, pcY;
    int pearsonX = 0, pearsonY = 0;

    # Prepare columns in X

    pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
    pcX.i = 0; pcX.n = ncx;

    if robustX
        # calculate threadPrepColBicor using data cptd in each thr
    else
        # calculate threadPrepColCor using data cptd in each thr

    # If the fallback method is to re-do everything in Pearson, check whether any columns had zero MAD.
    if (*fallback==3)
    {
      for (int t=0; t<nt; t++) if (thrdInfoX[t].zeroMAD > 0)
      {
        pearsonX = 1;
        if (*verbose)
          Rprintf("Warning in bicor(x, y): thread %d of %d reported zero MAD in column %d of x. %s",
                  t, nt, thrdInfoX[t].zeroMAD, "Switching to Pearson calculation for x.\n");
      }
      if (pearsonX==1) # Re-do all column preparations
      {
        for (int t = 0; t < nt; t++) thrdInfoX[t].fallback = 4;

        pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;
        pcX.i = 0; pcX.n = ncx;

        # calculate threadPrepColCor using data cptd in each thr
      }
    }


    # Prepare columns in Y

    # Rprintf(" ..preparing columns in y\n");
    pthread_mutex_t mutex1Y = PTHREAD_MUTEX_INITIALIZER;
    pcY.i = 0; pcY.n = ncy;

    if robustY
        # calculate threadPrepColBicor using data cptd in each thr
    else
        # calculate threadPrepColCor using data cptd in each thr

    # If the fallback method is to re-do everything in Pearson, check whether any columns had zero MAD.
    if (*fallback==3)
    {
      for (int t=0; t<nt; t++) if (thrdInfoY[t].zeroMAD > 0)
      {
        pearsonY = 1;
        if (*verbose)
          Rprintf("Warning in bicor(x, y): thread %d of %d reported zero MAD in column %d of y. %s",
                  t, nt, thrdInfoY[t].zeroMAD, "Switching to Pearson calculation for y.\n");
      }
      if (pearsonY==1) # Re-do all column preparations
      {
        for (int t = 0; t < nt; t++) thrdInfoY[t].fallback = 4;

        pthread_mutex_t mutex2Y = PTHREAD_MUTEX_INITIALIZER;
        pcY.i = 0;
        pcY.n = ncy;

        # calculate threadPrepColCor using data cptd in each thr
      }
    }


    result = multMatX' * multMatY

    # Remedial calculations

    # NA out all rows and columns that need it and check for values outside of [-1, 1]

    NA2ThreadData  natd[MxThreads];
    pcX.i = 0; pcY.i = 0;  # reset the progress counter
    pthread_t  thr2[MxThreads];

    for (int t=0; t<nt; t++)
    {
      natd[t].x = &thrdInfo[t];
      natd[t].pci = &pcX;
      natd[t].pcj = &pcY;
      status[t] = pthread_create_c(&thr2[t], NULL, threadNAing, (void *) &natd[t], thrdInfoX[t].threaded);
      if (status[t]!=0)
      {
         Rprintf("Error in bicor(x,y): thread %d could not be started successfully. Error code: %d.\n%s",
                 t, status[t], "*** WARNING: RETURNED RESULTS WILL BE INCORRECT. ***");
            *err = 2;
      }
    }

    for (int t=0; t<nt; t++)
       if (status[t]==0) pthread_join_c(thr2[t], NULL, thrdInfoX[t].threaded);


    nNA = nNA_ext

    # Clean up
    for (int t=nt-1; t >= 0; t--) free(aux[t]);
    free(NAmedY);
    free(NAmedX);
    free(nNAentriesY);
    free(nNAentriesX);
    free(multMatY);
    free(multMatX);
end

function cor(x::AbstractArray, y::AbstractArray, cosineX::Bool, cosineY::Bool,
             verbose::Bool)::Void
    """
    This can actually be relatively slow, since the search for calculations that need to be done is not
    parallel, so one thread may have to traverse the whole matrix. I can imagine parallelizing even that
    part, but for now leave it as is as this will at best be a minuscule improvement.
    """
    double * result, int *nNA
    assert(size(x, 1) == size(y, 1))
    nr = size(x, 1)
    ncx = size(x, 2)
    ncy = size(y, 2)
    size_t nNA_ext = 0

    # This matrix will hold preprocessed entries that can be simply multiplied together to get the
    # numerator
    multMatX = zeros(nr, ncx)
    multMatY = zeros(nr, ncy)
    # Number of NA entries in each column
    nNAentriesX = zeros(ncx)
    nNAentriesY = zeros(ncy)
    # Flag indicating whether the mean of each column is NA
    NAmeanX = falses(ncx)
    NAmeanY = falses(ncy)

    # Decide how many threads to use
    int nt = useNThreads( ncx* ncy, *nThreads);

    cor1ThreadData thrdInfoX[MxThreads];
    cor1ThreadData thrdInfoY[MxThreads];
    cor2ThreadData thrdInfo[MxThreads];

    # Prepare the multMat columns in X and Y
    cptd = Array{colPrepThreadData}(MxThreads)
    thr = Array{pthread_t}(MxThreads)
    status = Array{Int64}(MxThreads)  # normal: 0

    progressCounter pcX, pcY;

    # Prepare columns in X
    pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
    pcX.i = 0; pcX.n = ncx;
    # calculate threadPrepColCor using data cptd in each thr with thrdInfoX

    # Prepare columns in Y
    pthread_mutex_t mutex1Y = PTHREAD_MUTEX_INITIALIZER;
    pcY.i = 0; pcY.n = ncy;
    # calculate threadPrepColCor using data cptd in each thr with thrdInfoY

    result = multMatX' * multMatY

    # Remedial calculations

    # NA out all rows and columns that need it and check for values outside of [-1, 1]
    NA2ThreadData  natd[MxThreads];
    pcX.i = 0; pcY.i = 0;  # reset the progress counters
    pthread_t  thr2[MxThreads];

    for (int t=0; t<nt; t++)
    {
      natd[t].x = &thrdInfo[t];
      natd[t].pci = &pcX;
      natd[t].pcj = &pcY;
      status[t] = pthread_create_c(&thr2[t], NULL, threadNAing, (void *) &natd[t], thrdInfoX[t].threaded);
      if (status[t]!=0)
      {
         Rprintf("Error in cor(x,y): thread %d could not be started successfully. Error code: %d.\n%s",
                 t, status[t], "*** WARNING: RETURNED RESULTS WILL BE INCORRECT. ***");
            *err = 2;
      }
    }

    for (int t=0; t<nt; t++)
      if (status[t]==0)  pthread_join_c(thr2[t], NULL, thrdInfoX[t].threaded);

    nNA = nNA_ext;

    # clean up and return
    free(NAmeanY);
    free(NAmeanX);
    free(nNAentriesY);
    free(nNAentriesX);
    free(multMatY);
    free(multMatX);
end
