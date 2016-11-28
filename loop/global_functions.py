import cv2


def display_state(title, frame):
    #  title is a string that names the window, frame is the frame you want displayed
    cv2.imshow(title, frame)
    # TODO change self.current_state to frame downgraded
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()