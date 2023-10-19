boosters=(soft erlp ellipsoid)

nsets=(10 50 100 200 500 1000 10000)
nitems=(100)




for nset in ${nsets[@]} ; do
    for nitem in ${nitems[@]} ; do
        for booster in ${boosters[@]} ; do
            echo 'nset='$nset', nitem='$nitem', booster='$booster
        done
    done
done

