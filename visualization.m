for i = 0:29
    load(['./exp/cst_s_real/test/pred',sprintf('%04d',i),'.mat']);
    preds = [];
    truths = [];
    for j = 1:12
        preds = [preds,pred(:,:,j)];
        truths = [truths,truth(:,:,j)];
    end
    imwrite(double([preds;truths]),['./exp/cst_s_real/test/result',sprintf('%04d',i),'.tif']);
end