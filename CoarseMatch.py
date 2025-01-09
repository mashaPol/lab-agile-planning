import matplotlib.image as mpimage
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
from ImageFileUtils import LoadImage



if __name__ == "__main__":

    filenameOverview = "C:\\Work\\KundenData\\Daimler\\Bauteilbilder\\DL Teilekontrolle.jpg"
    filenameInspection1 = "C:\\Work\\KundenData\\Daimler\\Bauteilbilder\\DL\\K-2\\TK_DL_2.bmp"
    filenameInspection2 = "C:\\Work\\KundenData\\Daimler\\Bauteilbilder\\GPE\\K-1\\TK_GPE.bmp"

    overview = LoadImage(filenameOverview).getImage()

    inspection1 = LoadImage(filenameInspection1).getImage()
    inspection2 = LoadImage(filenameInspection2).getImage()



    w= inspection1.shape[0]
    h = inspection1.shape[1]
    found = False


    for scale in np.linspace(1.0, 0.1, 20)[::-1]:
        resized = imutils.resize(inspection1, width=int(w / scale))
        r = h / float(resized.shape[1])
        print("Scale: " + str(r))

        if ((resized.shape[0] > overview.shape[0]) or (resized.shape[1] > overview.shape[1])):
            continue

        res = cv2.matchTemplate(overview, inspection1, cv2.TM_CCORR_NORMED)
        threshold = 0.995
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(overview, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
            found = True

        if (found):
            break

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(overview)
    ax[0].axis('off')
    ax[1].imshow(inspection1)
    ax[1].axis('off')
    #ax[2].imshow(inspection2)
    plt.tight_layout()

    print("Match " + str(found) + "; at scale " + str(r))

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    plt.show(block=True)