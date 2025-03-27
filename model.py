from mediapipe import tasks
from classes import result
import numpy as np

RESULT = result()

def update_result(result, *args):
    global RESULT
    RESULT.landmarks[0].clear()
    RESULT.landmarks[1].clear()
    
    # print(len(result.handedness))
    # print(result.hand_landmarks)
    # print(result)

    for i in range(len(result.handedness)):
        if result.handedness[i][0].category_name == 'Left': # right (frame is flipped)
            idx = 1
        else: # left
            idx = 0
        # print(idx)
        
        RESULT.landmarks[idx] = result.hand_landmarks[i]
        if result.gestures[i][0].category_name != 'None':
            RESULT.gestures[idx] = result.gestures[i][0].category_name

        if len(RESULT.landmarks[idx]) == 21:
            # print(idx)
            thumb = RESULT.landmarks[idx][4]
            for i in range(2, 6): # [2, ..., 5]
                p = RESULT.landmarks[idx][i*4]
                if ((thumb.x - p.x)**2 + (thumb.y - p.y)**2)**0.5 < 0.05: # threshold = 0.05
                    RESULT.distances[idx][i-2] = True
                else:
                    RESULT.distances[idx][i-2] = False

            # thumb, index, middle, ring, pinky = RESULT.landmarks[idx][4], RESULT.landmarks[idx][8], RESULT.landmarks[idx][12], RESULT.landmarks[idx][16], RESULT.landmarks[idx][20]
            # RESULT.angles[idx] = np.arctan2(thumb.y - index.y, thumb.x - index.x) / np.pi
            # RESULT.distances[idx][0] = ((thumb.x - index.x)**2 + (thumb.y - index.y)**2)**0.5
            # RESULT.distances[idx][1] = ((thumb.x - middle.x)**2 + (thumb.y - middle.y)**2)**0.5
            # RESULT.distances[idx][2] = ((thumb.x - ring.x)**2 + (thumb.y - ring.y)**2)**0.5
            # RESULT.distances[idx][3] = ((thumb.x - pinky.x)**2 + (thumb.y - pinky.y)**2)**0.5
    # print(RESULT.landmarks)

options = tasks.vision.GestureRecognizerOptions(
    base_options=tasks.BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=tasks.vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=update_result,
    min_hand_detection_confidence = 0.8,
    min_hand_presence_confidence = 0.8,
    min_tracking_confidence = 0.5
)

recognizer = tasks.vision.GestureRecognizer.create_from_options(options)

'''
GestureRecognizerResult(
    gestures=[[Category(index=-1, score=0.9859821796417236, display_name='', category_name='None')]], 
    handedness=[[Category(index=0, score=0.6829532384872437, display_name='Right', category_name='Right')]], 
    hand_landmarks=[[
        NormalizedLandmark(x=0.04376596957445145, y=0.9778777360916138, z=1.7586305034456018e-07, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.13518676161766052, y=0.9115638732910156, z=-0.028059490025043488, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.20356996357440948, y=0.8177802562713623, z=-0.021471848711371422, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.24797096848487854, y=0.7365480065345764, z=-0.01312149129807949, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.26954057812690735, y=0.6578162312507629, z=-0.003889678744599223, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.11412688344717026, y=0.689222514629364, z=0.014329406432807446, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.12516234815120697, y=0.550223708152771, z=0.01993669755756855, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.12695571780204773, y=0.4799266457557678, z=0.020360929891467094, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.12473370134830475, y=0.4171193540096283, z=0.01944289542734623, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.04844246059656143, y=0.6941232681274414, z=0.019737612456083298, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.028721407055854797, y=0.5597122311592102, z=0.020280875265598297, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.01732097566127777, y=0.47380316257476807, z=0.0050306846387684345, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=0.006792128086090088, y=0.3985714614391327, z=-0.007000510115176439, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=-0.003007933497428894, y=0.7272143363952637, z=0.02009090781211853, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=-0.026698917150497437, y=0.6732285618782043, z=0.00269520515576005, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=-0.03526782989501953, y=0.701198935508728, z=-0.01917693391442299, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=-0.03747209906578064, y=0.7152520418167114, z=-0.02837550826370716, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=-0.0400751531124115, y=0.7705512046813965, z=0.01880059950053692, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=-0.06926548480987549, y=0.7290679216384888, z=0.005103512201458216, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=-0.08769024908542633, y=0.7337332963943481, z=-0.002036557998508215, visibility=0.0, presence=0.0), 
        NormalizedLandmark(x=-0.1000484824180603, y=0.7335342764854431, z=-0.004034693818539381, visibility=0.0, presence=0.0)]], 
    hand_world_landmarks=[[
        Landmark(x=-0.00787736102938652, y=0.073121078312397, z=0.02462412603199482, visibility=0.0, presence=0.0), 
        Landmark(x=0.022799598053097725, y=0.05350439250469208, z=0.01740645430982113, visibility=0.0, presence=0.0), 
        Landmark(x=0.04952939227223396, y=0.03582174703478813, z=0.014231900684535503, visibility=0.0, presence=0.0), 
        Landmark(x=0.06889007985591888, y=0.013197571039199829, z=0.007323392666876316, visibility=0.0, presence=0.0), 
        Landmark(x=0.07703982293605804, y=-0.007333017885684967, z=0.0014490068424493074, visibility=0.0, presence=0.0), 
        Landmark(x=0.020416071638464928, y=-0.009538323618471622, z=0.0064539797604084015, visibility=0.0, presence=0.0), 
        Landmark(x=0.02727026492357254, y=-0.03142111003398895, z=0.005230006296187639, visibility=0.0, presence=0.0), 
        Landmark(x=0.03182259947061539, y=-0.052100375294685364, z=0.00915420800447464, visibility=0.0, presence=0.0), 
        Landmark(x=0.031664490699768066, y=-0.06900431215763092, z=-0.0014477528166025877, visibility=0.0, presence=0.0), 
        Landmark(x=-8.753640577197075e-05, y=-0.005532621406018734, z=0.004082683008164167, visibility=0.0, presence=0.0), 
        Landmark(x=-0.001147191971540451, y=-0.03246888145804405, z=0.0007281876169145107, visibility=0.0, presence=0.0), 
        Landmark(x=-0.002682199701666832, y=-0.05405393987894058, z=-0.004666740074753761, visibility=0.0, presence=0.0), 
        Landmark(x=-0.0024936385452747345, y=-0.0740131363272667, z=-0.016940858215093613, visibility=0.0, presence=0.0), 
        Landmark(x=-0.014438921585679054, y=0.002567459363490343, z=-0.005670049227774143, visibility=0.0, presence=0.0), 
        Landmark(x=-0.01795104518532753, y=-0.008752190507948399, z=-0.012235857546329498, visibility=0.0, presence=0.0), 
        Landmark(x=-0.018751490861177444, y=-6.123073399066925e-05, z=-0.021877003833651543, visibility=0.0, presence=0.0), 
        Landmark(x=-0.014039076864719391, y=0.011109829880297184, z=-0.026535237208008766, visibility=0.0, presence=0.0), 
        Landmark(x=-0.030836071819067, y=0.017059700563549995, z=-0.01043503638356924, visibility=0.0, presence=0.0), 
        Landmark(x=-0.03409140184521675, y=0.008559935726225376, z=-0.013453339226543903, visibility=0.0, presence=0.0), 
        Landmark(x=-0.03709017485380173, y=0.008675407618284225, z=-0.013368651270866394, visibility=0.0, presence=0.0), 
        Landmark(x=-0.03874224051833153, y=0.006830526515841484, z=-0.017196353524923325, visibility=0.0, presence=0.0)]])
'''